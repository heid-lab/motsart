"""
Multi-head Flow Module for simultaneous R, TS, P geometry prediction.

Supports:
- Training on any combination of heads (R, TS, P)
- Configurable loss weights per head
- Different prior modes per head (rdkit, interpolation, pos_guess)
- Per-head metrics logging
"""

from typing import Dict, Optional, Callable, List, Tuple
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch
import lightning.pytorch as pl
from torchdiffeq import odeint
from pathlib import Path

from goflow.flow_matching.utils import (
    kabsch_align_batched,
    rmsd_loss,
    get_substruct_matches,
    get_min_dmae_match_torch_batch,
)
from goflow.test_samples_analysis import process_and_save_stats

from goflow.gotennet.models.components.outputs import Atomwise3DOut
from goflow.gotennet.models.representation.multihead_gotennet import MultiHeadGotenNet


class MultiHeadFlowModule(pl.LightningModule):
    """
    Multi-head Flow Matching module for R, TS, P prediction.

    Supports flexible training configurations:
    - Train on all heads simultaneously
    - Train on single head (e.g., TS only for benchmarking)
    - Configurable loss weights
    - Different prior strategies per head
    """

    def __init__(
            self,
            representation: MultiHeadGotenNet,
            lr: float = 5e-4,
            lr_decay: float = 0.5,
            lr_patience: int = 100,
            lr_minlr: float = 1e-6,
            lr_monitor: str = "validation/val_loss",

            task_name: str = "multihead_task",
            stats_dir: str = "reaction_analysis",

            weight_decay: float = 0.01,
            num_steps: int = 10,
            num_samples: int = 1,

            seed: int = 1,
            output: Optional[Dict] = None,
            scheduler: Optional[Callable] = None,
            lr_warmup_steps: int = 0,
            use_ema: bool = False,
            grad_clip_val: float = 1.0,

            # Multi-head specific params
            active_heads: List[str] = None,
            loss_weights: Dict[str, float] = None,
            noise_levels: Dict[str, float] = None,
            prior_modes: Dict[str, str] = None,
            # ODE solver for inference (torchdiffeq)
            ode_solver: str = "euler",

            **kwargs
    ):
        """
        Args:
            representation: MultiHeadGotenNet instance
            active_heads: List of heads to train ["R", "TS", "P"]
            loss_weights: Dict of loss weights per head {"R": 1.0, "TS": 1.0, "P": 1.0}
            noise_levels: Dict of noise levels per head for training
            prior_modes: Dict of prior modes per head {"R": "rdkit", "TS": "interpolation", "P": "rdkit"}
        """
        super().__init__()

        self.representation = representation

        # Default configurations
        if active_heads is None:
            active_heads = ["R", "TS", "P"]
        if loss_weights is None:
            loss_weights = {"R": 1.0, "TS": 1.0, "P": 1.0}
        if noise_levels is None:
            noise_levels = {"R": 0.4, "TS": 0.4, "P": 0.4}
        if prior_modes is None:
            prior_modes = {"R": "rdkit", "TS": "interpolation", "P": "rdkit"}

        self.active_heads = active_heads
        self.loss_weights = loss_weights
        self.noise_levels = noise_levels
        self.prior_modes = prior_modes
        # Create output heads for each active head
        self.output_heads = nn.ModuleDict({
            head: Atomwise3DOut(
                n_in=representation.hidden_dim,
                n_hidden=output['n_hidden'],
                activation=F.silu
            ) for head in active_heads
        })

        self.task_name = task_name
        self.stats_dir = Path(stats_dir)

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
        self.weight_decay = weight_decay

        self.num_steps = num_steps
        self.num_samples = num_samples

        self.use_ema = use_ema
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_minlr = lr_minlr
        self.grad_clip_val = grad_clip_val

        self.seed = seed
        self._original_seed = seed  # Store for reset during testing
        self.scheduler = scheduler

        self.ode_solver = ode_solver

        print(f"Multi-head Flow Module initialized")
        print(f"Active heads: {active_heads}")
        print(f"Loss weights: {loss_weights}")
        print(f"Prior modes: {prior_modes}")
        self.save_hyperparameters(ignore=['representation'])

        # Test results storage per head
        self.test_results = {head: [] for head in active_heads}
        self.train_steps = 0

    def _get_gt_pos(self, obj, head_name: str) -> Tensor:
        """Return pos_{head_name} if it exists, otherwise fall back to pos."""
        head_attr = f"pos_{head_name}"
        if hasattr(obj, head_attr):
            return getattr(obj, head_attr)
        return obj.pos

    def _sample_prior_per_head(
        self,
        head_name: str,
        gt_pos_N_3: Tensor,
        batch: Batch,
        return_clean: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Sample prior for a specific head.

        Prior strategies:
        - "rdkit": Sample from RDKit conformers (confs_R_N_M_3 or confs_P_N_M_3)
        - "interpolation": Interpolate between R and P conformers (for TS)
        - "pos_guess": Use provided geometry guess (for refinement tasks)
        - "gaussian": Sample from standard Gaussian N(0, I)

        Args:
            head_name: Name of the head ("R", "TS", or "P")
            gt_pos_N_3: Ground truth positions (used for shape reference)
            batch: PyG batch
            return_clean: If True, also return the clean prior (before noise) for init_cond conditioning

        Returns:
            Tuple of (x_0, init_cond) where:
                - x_0: Prior sample (with noise added during training for exact CFM)
                - init_cond: Clean prior (before noise) if return_clean=True, else None
        """
        prior_mode = self.prior_modes.get(head_name, "rdkit")
        noise_level = self.noise_levels.get(head_name, 0.4)

        if prior_mode == "gaussian":
            x_0 = torch.randn_like(gt_pos_N_3, device=self.device)
            # For gaussian prior, init_cond MUST be None to avoid data leakage.
            # If init_cond = x_0, then displacement = x_t - x_0 = t * dx_dt,
            # which trivially reveals the target velocity to the model.
            init_cond = None

        elif prior_mode == "pos_guess":
            # Use position guess for refinement
            pos_guess_attr = f"pos_guess_{head_name}"
            if hasattr(batch, pos_guess_attr):
                x_0_clean = getattr(batch, pos_guess_attr)
            elif hasattr(batch, "pos_guess"):
                x_0_clean = batch.pos_guess
            else:
                raise ValueError(f"pos_guess mode requires {pos_guess_attr} or pos_guess attribute in batch")

            init_cond = x_0_clean.clone() if return_clean else None

            if self.training:
                noise = torch.randn_like(x_0_clean) * noise_level
                x_0 = x_0_clean + noise
            else:
                x_0 = x_0_clean

        elif prior_mode == "rdkit":
            # Sample from RDKit conformers
            if head_name == "R":
                confs = batch.confs_R_N_M_3
            elif head_name == "P":
                confs = batch.confs_P_N_M_3
            else:
                raise ValueError(f"rdkit prior mode not supported for head {head_name}")

            # Randomly select one conformer per graph
            conf_select_G = torch.randint(0, confs.shape[1], (batch.num_graphs,), device=self.device)
            conf_select_N = conf_select_G[batch.batch]
            gather_indices = conf_select_N.view(-1, 1, 1).expand(-1, 1, 3)
            x_0_clean = torch.gather(confs, 1, gather_indices).squeeze(1)

            init_cond = x_0_clean.clone() if return_clean else None

            if self.training:
                noise = torch.randn_like(x_0_clean) * noise_level
                x_0 = x_0_clean + noise
            else:
                x_0 = x_0_clean

        elif prior_mode == "interpolation":
            # Interpolate between R and P conformers (for TS)
            # Use same conformer index for consistency
            n_confs_r = batch.confs_R_N_M_3.shape[1]
            n_confs_p = batch.confs_P_N_M_3.shape[1]
            assert n_confs_r <= n_confs_p, (
                f"R conformers ({n_confs_r}) exceeds P conformers ({n_confs_p}). "
                "Ensure preprocessing generates equal or more P conformers than R."
            )
            conf_select_G = torch.randint(0, n_confs_r, (batch.num_graphs,), device=self.device)
            conf_select_N = conf_select_G[batch.batch]
            gather_indices = conf_select_N.view(-1, 1, 1).expand(-1, 1, 3)

            r_conf = torch.gather(batch.confs_R_N_M_3, 1, gather_indices).squeeze(1)
            p_conf = torch.gather(batch.confs_P_N_M_3, 1, gather_indices).squeeze(1)

            # Align P to R first
            p_aligned = kabsch_align_batched(r_conf, p_conf, batch.batch)

            # Interpolate (midpoint)
            x_0_clean = 0.5 * (r_conf + p_aligned)

            init_cond = x_0_clean.clone() if return_clean else None

            if self.training:
                noise = torch.randn_like(x_0_clean) * noise_level
                x_0 = x_0_clean + noise
            else:
                x_0 = x_0_clean

        else:
            raise ValueError(f"Unknown prior mode: {prior_mode}")

        return x_0, init_cond

    def get_perturbed_flow_point_and_time(
        self,
        batch: Batch,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor, Optional[Dict[str, Tensor]], Optional[Dict[str, Tensor]]]:
        """
        Create perturbed points and velocity targets for all active heads.

        Uses the SAME time t for all heads in a single forward pass.

        Returns:
            x_t_dict: Dict of interpolated positions per head
            dx_dt_dict: Dict of velocity targets per head
            t_G: Time per graph (shared)
            init_cond_dict: Dict of initial condition positions (clean priors) per head,
                           or None if use_init_cond is disabled in representation
            s_target_dict: None (kept for API compatibility)
        """
        use_init_cond = getattr(self.representation, 'use_init_cond', False)

        t_G = torch.rand(batch.num_graphs, 1, device=self.device)
        t_N = t_G[batch.batch]

        x_t_dict = {}
        dx_dt_dict = {}
        init_cond_dict = {} if use_init_cond else None

        for head_name in self.active_heads:
            x_1 = self._get_gt_pos(batch, head_name)
            x_0, init_cond = self._sample_prior_per_head(
                head_name, x_1, batch, return_clean=use_init_cond
            )
            if init_cond_dict is not None and init_cond is not None:
                init_cond_dict[head_name] = init_cond

            x_1_aligned = kabsch_align_batched(x_0, x_1, batch.batch)
            x_t = (1 - t_N) * x_0 + t_N * x_1_aligned
            dx_dt = x_1_aligned - x_0

            x_t_dict[head_name] = x_t
            dx_dt_dict[head_name] = dx_dt

        return x_t_dict, dx_dt_dict, t_G, init_cond_dict, None

    def model_output(
        self,
        x_t_dict: Dict[str, Tensor],
        batch: Batch,
        t_G: Tensor,
        init_cond_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], None]:
        """
        Get velocity predictions for all active heads.

        Args:
            x_t_dict: Dict mapping head names to position tensors
            batch: PyTorch Geometric batch
            t_G: Time per graph
            init_cond_dict: Optional dict mapping head names to initial condition positions (t=0 prior).
                           Used for FiLM conditioning when representation.use_init_cond=True.

        Returns:
            Tuple of:
                - Dict mapping head names to predicted velocities
                - None (kept for API compatibility)
        """
        # Forward through multi-head representation
        features_dict = self.representation(
            x_t_dict, t_G, batch, init_cond_dict=init_cond_dict
        )

        # Compute velocity outputs
        velocity_outputs = {}
        for head_name in self.active_heads:
            h_N_D, X_N_L_D = features_dict[head_name]
            velocity_outputs[head_name] = self.output_heads[head_name](h_N_D, X_N_L_D[:, :3, :])

        return velocity_outputs, None

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        optimizer_closure()

        if self.grad_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        """Training step with multi-head flow matching loss."""
        x_t_dict, dx_dt_dict, t_G, init_cond_dict, _ = \
            self.get_perturbed_flow_point_and_time(batch)

        pred_dict, _ = self.model_output(
            x_t_dict, batch, t_G,
            init_cond_dict=init_cond_dict,
        )

        total_loss = torch.tensor(0.0, device=self.device)

        for head_name in self.active_heads:
            # RMSD loss for this head
            loss_head = rmsd_loss(pred_dict[head_name], dx_dt_dict[head_name])
            weighted_loss = self.loss_weights.get(head_name, 1.0) * loss_head
            total_loss = total_loss + weighted_loss

            # Log per-head metrics
            self.log(
                f"train/rmsd_{head_name}",
                loss_head,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch.num_graphs
            )

        self.log(
            "train/total_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs
        )

        return total_loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
        """Validation step with multi-head metrics."""
        x_t_dict, dx_dt_dict, t_G, init_cond_dict, _ = \
            self.get_perturbed_flow_point_and_time(batch)
        pred_dict, _ = self.model_output(
            x_t_dict, batch, t_G,
            init_cond_dict=init_cond_dict,
        )

        fm_total_loss = torch.tensor(0.0, device=self.device)

        for head_name in self.active_heads:
            loss_head = rmsd_loss(pred_dict[head_name], dx_dt_dict[head_name])
            weighted_loss = self.loss_weights.get(head_name, 1.0) * loss_head
            fm_total_loss = fm_total_loss + weighted_loss

            self.log(
                f"validation/rmsd_{head_name}",
                loss_head,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch.num_graphs
            )

        self.log(
            "validation/val_loss",
            fm_total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs
        )

        return fm_total_loss

    def test_step(self, batch: Batch, batch_idx: int):
        """Generate samples for all heads via joint ODE integration.

        IMPORTANT: All active heads are integrated jointly in a single ODE solve,
        matching the training behavior where all heads are processed together.
        This ensures cross-attention and shared representations work correctly.
        """
        self.seed += 1
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        t_T = torch.linspace(0, 1, steps=self.num_steps, device=self.device)

        # Generate samples with joint integration of all heads
        self._test_step_joint(batch, t_T)

    def _test_step_joint(self, batch: Batch, t_T: Tensor):
        """Generate samples for all heads via joint ODE integration.

        Integrates all active heads together in a single ODE solve from t=0 to t=1.
        init_cond is sampled ONCE and kept FIXED throughout the integration.
        """
        N = batch.num_nodes
        use_init_cond = getattr(self.representation, 'use_init_cond', False)

        # Storage for all samples per head: {head_name: (S, N, 3)}
        pos_gen_per_head = {
            head: torch.zeros((self.num_samples, N, 3), device=self.device)
            for head in self.active_heads
        }

        for sample_idx in range(self.num_samples):
            torch.manual_seed(self.seed + sample_idx)

            # Sample priors for ALL heads
            initial_states = []
            init_cond_dict = {} if use_init_cond else None

            for head_name in self.active_heads:
                gt_pos = self._get_gt_pos(batch, head_name)
                pos_init, init_cond = self._sample_prior_per_head(
                    head_name, gt_pos, batch, return_clean=use_init_cond
                )
                initial_states.append(pos_init)

                if init_cond_dict is not None and init_cond is not None:
                    init_cond_dict[head_name] = init_cond

            # Stack into joint state: (num_heads, N, 3)
            state_init = torch.stack(initial_states, dim=0)

            # Closure for ODE func
            _init_cond_dict = init_cond_dict

            def ode_func(t, state_H_N_3: Tensor, _icd=_init_cond_dict) -> Tensor:
                t_scalar = t.item() if isinstance(t, torch.Tensor) else float(t)
                t_G = torch.full(
                    (batch.num_graphs, 1), t_scalar,
                    device=self.device, dtype=state_H_N_3.dtype
                )
                x_t_dict = {head: state_H_N_3[i] for i, head in enumerate(self.active_heads)}
                features_dict = self.representation(x_t_dict, t_G, batch, init_cond_dict=_icd)
                velocities = []
                for head_name in self.active_heads:
                    h_N_D, X_N_L_D = features_dict[head_name]
                    vel = self.output_heads[head_name](h_N_D, X_N_L_D[:, :3, :])
                    velocities.append(vel)
                return torch.stack(velocities, dim=0)

            trajectory = odeint(ode_func, state_init, t_T, method=self.ode_solver)
            final_state = trajectory[-1]

            # Log RMSD to GT
            for i, head_name in enumerate(self.active_heads):
                pred = final_state[i].detach()
                gt_pos = self._get_gt_pos(batch, head_name)
                if gt_pos is not None:
                    gt_aligned = kabsch_align_batched(pred, gt_pos, batch.batch)
                    rmsd = torch.sqrt(torch.mean((pred - gt_aligned) ** 2)).item()
                    print(f"  [test] sample={sample_idx} {head_name}: RMSD={rmsd:.4f}")

            # Store results per head
            for i, head_name in enumerate(self.active_heads):
                pos_gen_per_head[head_name][sample_idx] = final_state[i]

        # Process per-molecule results for each head
        # Each head gets its own data list to match original behavior
        for head_name in self.active_heads:
            pos_gen_S_N_3 = pos_gen_per_head[head_name]

            for j, data in enumerate(batch.to_data_list()):
                mask = (batch.batch == j).cpu()
                pos_gen_S_Nm_3 = pos_gen_S_N_3[:, mask]

                if torch.isinf(pos_gen_S_Nm_3).any():
                    print(f"Infinite value in pos encountered for head {head_name}!")
                    continue

                # Get ground truth for this molecule (pos_{head} or pos)
                gt_pos_mol = self._get_gt_pos(data, head_name)
                if gt_pos_mol is not None:
                    pos_gen_S_Nm_3 = self._substruct_match_and_kabsch_align(
                        data, pos_gen_S_Nm_3, gt_pos_mol
                    )

                # Aggregate samples: pick median-closest
                if self.num_samples > 1:
                    pos_aggr = torch.median(pos_gen_S_Nm_3, dim=0).values
                    distances = torch.linalg.vector_norm(pos_gen_S_Nm_3 - pos_aggr, dim=(1, 2))
                    pos_best = pos_gen_S_Nm_3[torch.argmin(distances)]
                else:
                    pos_best = pos_gen_S_Nm_3[0]

                # Store results
                setattr(data, f"pos_gen_{head_name}", pos_best)
                setattr(data, f"pos_gen_all_samples_{head_name}", pos_gen_S_Nm_3)

                self.test_results[head_name].append(data.to("cpu"))

    def _substruct_match_and_kabsch_align(
        self,
        data,
        pos_gen_S_Nm_3: Tensor,
        pos_gt_Nm_3: Tensor
    ) -> Tensor:
        """Substructure match and Kabsch align samples to ground truth."""
        matches_M_N = get_substruct_matches(data.smiles)
        match_S_Nm = get_min_dmae_match_torch_batch(matches_M_N, pos_gt_Nm_3, pos_gen_S_Nm_3)
        pos_gen_S_Nm_3 = torch.gather(
            pos_gen_S_Nm_3, 1,
            match_S_Nm.unsqueeze(-1).expand(-1, -1, 3)
        )

        S = pos_gen_S_Nm_3.shape[0]
        Nm = pos_gen_S_Nm_3.shape[1]

        pos_gt_SNm_3 = pos_gt_Nm_3.repeat(S, 1)
        pos_gen_SNm_3 = pos_gen_S_Nm_3.reshape(S * Nm, 3)

        batch_idx = torch.arange(S, device=self.device).repeat_interleave(Nm)
        pos_gen_aligned = kabsch_align_batched(pos_gt_SNm_3, pos_gen_SNm_3, batch_idx)

        return pos_gen_aligned.reshape(S, Nm, 3)

    def on_test_epoch_start(self):
        """Initialize test results storage and reset seed for reproducibility."""
        self.test_results = {head: [] for head in self.active_heads}
        self.seed = self._original_seed  # Reset seed for reproducible testing

    def on_test_epoch_end(self):
        """Process and save test statistics for each head."""
        for head_name in self.active_heads:
            if self.test_results[head_name]:
                # Map head-specific attributes to expected names for process_and_save_stats
                # process_and_save_stats expects: pos_gen, pos_gen_all_samples_S_N_3, pos (ground truth)
                for data in self.test_results[head_name]:
                    data.pos_gen = getattr(data, f"pos_gen_{head_name}")
                    data.pos_gen_all_samples_S_N_3 = getattr(data, f"pos_gen_all_samples_{head_name}")
                    data.pos = self._get_gt_pos(data, head_name)
                out_path = self.stats_dir / f"{self.task_name}_{head_name}"
                process_and_save_stats(self.test_results[head_name], out_path=out_path)
                print(f"Saved test stats for head {head_name} to {out_path}")

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-7,
        )

        if self.scheduler and callable(self.scheduler):
            print(f"Using custom scheduler: {self.scheduler}")
            scheduler, _ = self.scheduler(optimizer=optimizer)
        else:
            print(f"Using default ReduceLROnPlateau scheduler (patience={self.lr_patience}, factor={self.lr_decay})")
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
