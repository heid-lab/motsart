from typing import Dict, Optional, Callable, List, Tuple
import numpy as np
import torch
from torch_geometric.nn import radius_graph
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
from torchdiffeq import odeint
from pathlib import Path

from goflow.flow_matching.utils import (
    kabsch_align_batched,
    rmsd_loss,
    get_substruct_matches, 
    get_min_dmae_match_torch_batch,
    generate_smiles_conformers,
)
from goflow.test_samples_analysis import process_and_save_stats

from goflow.gotennet.models.components.outputs import Atomwise3DOut
from goflow.gotennet.models.representation.gotennet import GotenNet

import time

class GraphTimeMLP(nn.Module):
    """
    MLP that takes per-graph times t_G and outputs per-graph coefficients.
    """
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, t_G):
        # t_G: (num_graphs,) or (num_graphs, 1)
        if t_G.dim() == 1:
            t_G = t_G.unsqueeze(-1)
        coeffs = self.mlp(t_G)  # (num_graphs, 1)
        return coeffs.squeeze(-1)  # (num_graphs,)

def expand_coeffs_to_nodes(coeffs, batch):
    """
    Expand per-graph coefficients to per-node coefficients.
    Args:
        coeffs: (num_graphs,)
        batch: (num_nodes,) tensor mapping each node to its graph index
    Returns:
        per_node_coeffs: (num_nodes,)
    """
    return coeffs[batch]

class FlowModule(pl.LightningModule):
    def __init__(
            self,
            representation: GotenNet,
            lr: float = 5e-4,
            lr_decay: float = 0.5,
            lr_patience: int = 100,
            lr_minlr: float = 1e-6,
            lr_monitor: str = "validation/ema_val_loss",
            
            task_name: str = "task_name_default",
            stats_dir: str = "reaction_analysis",
            
            weight_decay: float = 0.01,
            num_steps: int = 10,
            num_samples: int = 1,
            
            use_bond_loss: bool = False,
            comb_rmsd_energy_loss: bool = True,
            
            seed: int = 1,
            output: Optional[Dict] = None,
            scheduler: Optional[Callable] = None,
            lr_warmup_steps: int = 0,
            use_ema: bool = False,
            sample_method: str = "gaussian",
            noise_level: float = 0.4,
            ode_solver: str = "euler",
            **kwargs
    ):
        super().__init__()
        self.representation = representation
        self.atomwise_3D_out_layer = Atomwise3DOut(n_in=representation.hidden_dim, n_hidden=output['n_hidden'], activation=F.silu)

        self.task_name = task_name
        self.stats_dir = Path(stats_dir)
        
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
        self.weight_decay = weight_decay

        self.num_steps = num_steps
        self.num_samples = num_samples
        
        self.use_bond_loss = use_bond_loss
        self.comb_rmsd_energy_loss = comb_rmsd_energy_loss

        self.use_ema = use_ema
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_minlr = lr_minlr

        self.seed = seed

        self.scheduler = scheduler
        self.sample_method = sample_method
        self.noise_level = noise_level
        self.ode_solver = ode_solver

        print("FM seed", self.seed)
        print(f"Sample method: {sample_method}")

        self.save_hyperparameters(ignore=['representation'])

        self.test_results_C = []  # save results in test_step
        self.train_steps = 0

        #self.conformers_per_smiles_in_cache = 12
        #self.smiles_to_conformers_cache = {}


    """def generate_rdkit_conformers(self, smiles_list: List[str], r_or_p=0) -> List[np.ndarray]:
        coords_list: List[np.ndarray] = []
        
        for smiles in smiles_list:
            if smiles in self.smiles_to_conformers_cache:
            else:
                coords_M_N_3 = generate_smiles_conformers(smiles.split('>>')[r_or_p], n_confs=self.conformers_per_smiles_in_cache)
                self.smiles_to_conformers_cache[smiles] = co

            coords_list.append(coords_N_3)
        return coords_list"""
    
    def _sample_prior(self, x_1_N_3, batch):
        if self.sample_method == "gaussian":
            x_0_N_3 = torch.randn_like(x_1_N_3, device=self.device)
        elif self.sample_method == "pos_guess":
            x_0_N_3 = batch.pos_guess
            if self.training:
                noise_N_3 = torch.randn_like(x_0_N_3) * 0.1
                x_0_N_3 = x_0_N_3 + noise_N_3
        elif self.sample_method == "rdkit_reactant":
            conf_select_G = torch.randint(0, batch.confs_N_M_3.shape[1], (batch.num_graphs,)).to(self.device)
            conf_select_N = conf_select_G[batch.batch]
            
            gather_indices_N_1_3 = conf_select_N.view(-1, 1, 1).expand(-1, 1, 3)
            confs_N_3 = torch.gather(batch.confs_N_M_3, 1, gather_indices_N_1_3).squeeze(1)
            
            x_0_N_3 = confs_N_3
            if self.training:
                noise_N_3 = torch.randn_like(x_1_N_3) * self.noise_level
                x_0_N_3 = x_0_N_3 + noise_N_3
        else:
            raise NotImplementedError("Either choose gaussian or pos_guess as sample_method in config")

        return x_0_N_3

    def get_perturbed_flow_point_and_time(self, batch: Batch, t_lower: float = 0):
        x_1_N_3 = batch.pos
        x_0_N_3 = self._sample_prior(x_1_N_3, batch)
        
        t_G = torch.rand(batch.num_graphs, 1, device=self.device) * (1 - t_lower) + t_lower
        t_N = t_G[batch.batch]

        x_1_aligned_N_3 = kabsch_align_batched(x_0_N_3, x_1_N_3, batch.batch)
        x_t_N_3 = (1 - t_N) * x_0_N_3 + t_N * x_1_aligned_N_3
        dx_dt_N_3 = x_1_aligned_N_3 - x_0_N_3

        return x_t_N_3, dx_dt_N_3, t_G


    def model_output(self, x_t_N_3, batch: Batch, t_G: Tensor) -> Tensor:
        h_N_D, X_N_L_D = self.representation(x_t_N_3, t_G, batch)
        atom_N_3 = self.atomwise_3D_out_layer(h_N_D, X_N_L_D[:, :3, :])
        return atom_N_3


    def get_bond_loss(self, batch, x1_hat_N_3, t_N):
        t_lower = .9

        # 2. Proper scaling: c(t)^-1 = (1-t)^-2
        # This aligns the target-based penalty with the velocity-based Flow Matching scale.
        # We use a larger eps (1e-2) to prevent the loss from exploding as t -> 1.
        #t_E = t_N[batch.edge_index[0]].squeeze() TODO: figure out
        #eps_t = 1e-3
        #scaling_E = 1.0 / (torch.square(1.0 - t_E) + eps_t)
        
        # Only add bond loss on t > t_lower. Makes training more stable, since structures are closer to GT.
        t_lower_mask_N = t_N > t_lower
        t_lower_mask_E = t_lower_mask_N[batch.edge_index[0]].squeeze()

        x1_hat_b1_E_3 = x1_hat_N_3[batch.edge_index[0]]
        x1_hat_b2_E_3 = x1_hat_N_3[batch.edge_index[1]]
        x1_hat_dist_E = torch.linalg.vector_norm(x1_hat_b1_E_3 - x1_hat_b2_E_3, dim=1)

        x1_gt_b1_E_3 = batch.pos[batch.edge_index[0]]
        x1_gt_b2_E_3 = batch.pos[batch.edge_index[1]]
        x1_gt_dist_E = torch.linalg.vector_norm(x1_gt_b1_E_3 - x1_gt_b2_E_3, dim=1)

        eps = 1e-9
        penalty_E = torch.log((x1_hat_dist_E + eps) / (x1_gt_dist_E + eps)) ** 2 # * scaling_E
        penalty_Ef = penalty_E[t_lower_mask_E]

        if penalty_Ef.numel() > 0:
            return penalty_Ef.mean()
        else:
            return torch.tensor(0.0, device=x1_hat_N_3.device, requires_grad=True)

        
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:        
        x_t_N_3, dx_dt_N_3, t_G = self.get_perturbed_flow_point_and_time(batch)
        diff_vec_N_3 = self.model_output(x_t_N_3, batch, t_G)

        loss = rmsd_loss(diff_vec_N_3, dx_dt_N_3)
        rmsd = loss.item()
        
        t_N = t_G[batch.batch]
        x1_hat_N_3 = x_t_N_3 + (1-t_N) * diff_vec_N_3
        
        if self.use_bond_loss:
            loss_bond = self.get_bond_loss(batch, x1_hat_N_3, t_N)
            loss = loss + 1e2 * loss_bond
            print(f"Bond loss: {loss_bond.item()}")            
            self.log("train/loss_bond", loss_bond, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
                
        self.log("train/rmsd", rmsd, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
        
        return loss


    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
        x_t_N_3, dx_dt_N_3, t_G = self.get_perturbed_flow_point_and_time(batch)
        diff_vec_N_3 = self.model_output(x_t_N_3, batch, t_G)
        
        loss = rmsd_loss(diff_vec_N_3, dx_dt_N_3)
        rmsd = loss.item()

        t_N = t_G[batch.batch]        
        x1_hat_N_3 = x_t_N_3 + (1-t_N) * diff_vec_N_3
        
        loss_bond = self.get_bond_loss(batch, x1_hat_N_3, t_N)
        self.log("validation/loss_bond", loss_bond, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
        self.log("validation/val_loss", rmsd, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
        
        return loss

    # S... number of samples
    def test_step(self, batch: Batch, batch_idx: int):
        self.seed += 1
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        t_T = torch.linspace(0, 1, steps=self.num_steps, device=self.device)

        def ode_func(t, x_t_N_3):
            t_G = torch.tensor([t] * batch.num_graphs, device=self.device)
            model_forces_N_3 = self.model_output(x_t_N_3, batch, t_G)
            return model_forces_N_3

        # Generate num_samples trajectories for batch
        pos_gen_S_N_3 = torch.zeros((self.num_samples, batch.num_nodes, 3), device=self.device)
        
        for i in range(self.num_samples):
            if self.seed is not None:
                torch.manual_seed(self.seed + i)
            pos_init_N_3 = self._sample_prior(batch.pos, batch)
            pos_gen_S_N_3[i, ...] = odeint(ode_func, pos_init_N_3, t_T, method=self.ode_solver)[-1]

        pos_gen_C_Nm_3 = []
        for j, data in enumerate(batch.to_data_list()):
            # Get single molecule positions from sampled trajectories
            mask = (batch.batch == j).cpu()
            pos_gen_S_Nm_3 = pos_gen_S_N_3[:, mask]
            if torch.isinf(pos_gen_S_Nm_3).any(): 
                print("Infinite value in pos encountered!")
                continue
            
            # If ground-truth pos exists, match and align samples to it
            if data.pos is not None:
                pos_gen_S_Nm_3 = self.substruct_match_and_kabsch_align_samples(data, pos_gen_S_Nm_3)

            # -------------------------- START: Aggregate the S samples --------------------------
            if self.num_samples > 1:
                pos_aggr_Nm_3 = torch.median(pos_gen_S_Nm_3, dim=0).values
                distances_S = torch.linalg.vector_norm(pos_gen_S_Nm_3 - pos_aggr_Nm_3, dim=(1, 2))
                pos_best_Nm_3 = pos_gen_S_Nm_3[torch.argmin(distances_S)]
            else:
                assert len(pos_gen_S_Nm_3) == 1
                pos_best_Nm_3 = pos_gen_S_Nm_3[0]
            # -------------------------- END: Aggregate the S samples --------------------------

            data.pos_gen = pos_best_Nm_3
            data.pos_gen_all_samples_S_N_3 = pos_gen_S_Nm_3
            pos_gen_C_Nm_3.append(pos_best_Nm_3)
            self.test_results_C.append(data.to("cpu"))
        
    def on_test_epoch_start(self):
        self.test_results_C = []

    def on_test_epoch_end(self):
        process_and_save_stats(self.test_results_C, out_path=self.stats_dir / self.task_name)
        
    def substruct_match_and_kabsch_align_samples(self, data, pos_gen_S_Nm_3):
        pos_gt_Nm_3 = data.pos
        
        # Substructure matching (batched for S)
        matches_M_N = get_substruct_matches(data.smiles)
        match_S_Nm = get_min_dmae_match_torch_batch(matches_M_N, pos_gt_Nm_3, pos_gen_S_Nm_3)
        pos_gen_S_Nm_3 = torch.gather(pos_gen_S_Nm_3, 1, match_S_Nm.unsqueeze(-1).expand(-1,-1,3))
        
        # Kabsch rotation
        S = pos_gen_S_Nm_3.shape[0]
        Nm = pos_gen_S_Nm_3.shape[1]
        
        # This is a trick to make the batched rotation to the GT molecule easy
        # Repeat GT pos S times (have to rotate each sample to it)
        pos_gt_SNm_3 = pos_gt_Nm_3.repeat(S, 1)
        pos_gen_SNm_3 = pos_gen_S_Nm_3.reshape(S*Nm, 3)
        
        batch = torch.arange(S, device=self.device).repeat_interleave(Nm)
        pos_gen_aligned_SNm_3 = kabsch_align_batched(pos_gt_SNm_3, pos_gen_SNm_3, batch)
        return pos_gen_aligned_SNm_3.reshape(S, Nm, 3)


    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure optimizers and learning rate schedulers."""
        print("self.weight_decay", self.weight_decay)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-7,
        )

        if self.scheduler and callable(self.scheduler):
            scheduler, _ = self.scheduler(optimizer=optimizer)
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                min_lr=self.lr_minlr,
            )

        schedule = {
            "scheduler": scheduler,
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }

        return [optimizer], [schedule]
