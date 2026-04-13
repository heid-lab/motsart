"""
RTSP Guesser: Generate Reactant, TS, and Product geometries using MultiHead GoFlow.

This module uses a trained MultiHeadFlowModule to predict R, TS, and P geometries
from reaction SMILES. The prior at t=0 is:
- R: RDKit-generated conformers
- TS: Interpolation between R and P conformers
- P: RDKit-generated conformers

The ODE is integrated jointly for all heads to produce final geometries.

Usage:
    python -m motsart.learning.rtsp_guesser env=local env.rxn_num=0
"""

import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from hydra_zen import zen, store

from goflow.preprocessing import process_reaction_data
from goflow.flow_matching.utils import generate_smiles_conformers
from goflow.flow_matching.multihead_flow_module import MultiHeadFlowModule

from motsart.conf_default import EnvironmentConfig, ALConfig
from motsart.common import PathHandler
from motsart.complex_finder.utils import write_xyz_from_tensor


class RTSPGuesser:
    """
    Generate R, TS, P geometries using MultiHeadFlowModule.

    Starting from RDKit conformers (for R and P) and their interpolation (for TS),
    integrates the flow ODE to produce optimized geometries.
    """

    def __init__(
        self,
        multihead_module: MultiHeadFlowModule,
        rxn_id: str,
        rxn_smiles: str,
        results_folder: str = 'results',
        n_conformers: int = 32,
        num_samples: int = 3,
    ):
        self.rxn_id = rxn_id
        self.rxn_smiles = rxn_smiles
        self.n_conformers = n_conformers
        self.num_samples = num_samples

        # PathHandler for saving outputs
        # Use 'rtsp_goflow' as the ts_method for TS outputs
        self.path_handler = PathHandler(
            rxn_id=rxn_id,
            r_or_p='r',
            ts_method='rtsp_goflow',
            results_folder=results_folder
        )

        # Create output directories
        self.path_handler.rp_dir_final.mkdir(parents=True, exist_ok=True)
        self.path_handler.p_dir.mkdir(parents=True, exist_ok=True)
        self.path_handler.ts_to_validate.mkdir(parents=True, exist_ok=True)

        self.load_and_set_inference_module(multihead_module)

    def load_and_set_inference_module(self, multihead_module: MultiHeadFlowModule):
        """Load checkpoint and prepare module for inference."""
        with open(self.path_handler.learning_feat_dict, 'rb') as f:
            self.feat_dict = pickle.load(f)

        # Load checkpoint - use a multihead-specific checkpoint path
        ckpt_path = self.path_handler.learning_dir / 'goflow_multihead.ckpt'
        if not ckpt_path.exists():
            # Fallback to default checkpoint
            ckpt_path = self.path_handler.learning_ckpt

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        multihead_module.load_state_dict(state_dict)
        multihead_module.eval()
        self.multihead_module = multihead_module

        # Set num_samples from config
        self.multihead_module.num_samples = self.num_samples

    def generate_conformers(self):
        """Generate RDKit conformers for R and P from reaction SMILES."""
        r_smi, p_smi = self.rxn_smiles.split(">>")

        r_confs = generate_smiles_conformers(r_smi, self.n_conformers)
        p_confs = generate_smiles_conformers(p_smi, self.n_conformers)

        if r_confs is None or p_confs is None:
            raise ValueError(f"Failed to generate conformers for reaction {self.rxn_id}")

        # Shape from generate_smiles_conformers: (M_conformers, N_atoms, 3)
        # Transpose to (N_atoms, M_conformers, 3) as expected by MultiHeadFlowModule
        self.confs_R_N_M_3 = torch.tensor(r_confs).float().transpose(0, 1).contiguous()
        self.confs_P_N_M_3 = torch.tensor(p_confs).float().transpose(0, 1).contiguous()

        return self.confs_R_N_M_3, self.confs_P_N_M_3

    def create_data_object(self):
        """Create PyG data object for inference."""
        # Generate conformers
        confs_R, confs_P = self.generate_conformers()

        # Use first R conformer as reference geometry for graph construction
        ref_pos = confs_R[:, 0, :].numpy()  # (N, 3)

        # Create data object using process_reaction_data
        data_obj = process_reaction_data(
            feat_dict=self.feat_dict,
            rxn_smiles=self.rxn_smiles,
            rxn_id=self.rxn_id,
            gt_xyzs_C_N_3=[ref_pos],  # Use as reference for graph structure
        )[0]

        # Add conformers for prior sampling
        data_obj.confs_R_N_M_3 = confs_R
        data_obj.confs_P_N_M_3 = confs_P

        # Add placeholder ground truth positions (needed by _sample_prior_per_head for shape reference)
        # Use first conformers as placeholders since we don't have ground truth
        data_obj.pos_R = confs_R[:, 0, :].clone()
        data_obj.pos_P = confs_P[:, 0, :].clone()
        # TS prior is interpolation, so use midpoint of R and P
        data_obj.pos_TS = 0.5 * (data_obj.pos_R + data_obj.pos_P)
        data_obj.pos = data_obj.pos_TS.clone()  # For backward compatibility

        # Store metadata
        data_obj.rxn_index = self.rxn_id

        return data_obj

    def run_inference(self):
        """Run inference and save R, TS, P geometries."""
        print(f"Running RTSP inference on rxn {self.rxn_id}")

        # Create data object
        data_obj = self.create_data_object()

        # Move to device
        device = next(self.multihead_module.parameters()).device
        data_obj = data_obj.to(device)
        batch = Batch.from_data_list([data_obj])

        # Run inference
        with torch.inference_mode():
            # Reset test results storage
            self.multihead_module.on_test_epoch_start()

            # Run test step (generates samples via ODE integration)
            self.multihead_module.test_step(batch, 0)

        # Extract and save results
        self._save_results()

        print(f"Saved generated xyz files to:")
        print(f"  R: {self.path_handler.rp_dir_final}")
        print(f"  P: {self.path_handler.p_dir}")
        print(f"  TS: {self.path_handler.ts_to_validate}")

    def _save_results(self):
        """Save generated geometries to xyz files."""
        # Get atom types from first result (same for all heads)
        first_head = self.multihead_module.active_heads[0]
        if not self.multihead_module.test_results[first_head]:
            raise RuntimeError("No test results generated")

        data = self.multihead_module.test_results[first_head][0]
        atom_type = data.atom_type

        # Save each sample for each head
        for head_name in self.multihead_module.active_heads:
            result_data = self.multihead_module.test_results[head_name][0]

            # Get all samples: (S, N, 3)
            pos_all_samples = getattr(result_data, f"pos_gen_all_samples_{head_name}")

            # Determine output directory based on head
            if head_name == "R":
                out_dir = self.path_handler.rp_dir_final
            elif head_name == "P":
                out_dir = self.path_handler.p_dir
            elif head_name == "TS":
                out_dir = self.path_handler.ts_to_validate
            else:
                continue

            # Save each sample as separate xyz file
            for sample_idx in range(pos_all_samples.shape[0]):
                pos = pos_all_samples[sample_idx]  # (N, 3)
                mol_name = f"mol_{sample_idx:03d}.xyz"
                out_path = out_dir / mol_name
                write_xyz_from_tensor(atom_type, pos, out_path)


def rtsp_guesser_task(
    multihead_module: MultiHeadFlowModule,
    env: EnvironmentConfig,
    al_cfg: ALConfig,
):
    """Main task function for RTSP guessing."""
    df_smi = pd.read_csv(env.rxn_csv, sep=',', header=None)
    rxn_id = df_smi[0].values[env.rxn_num]
    rxn_smiles = df_smi[1].values[env.rxn_num]

    guesser = RTSPGuesser(
        multihead_module=multihead_module,
        rxn_id=str(rxn_id),
        rxn_smiles=rxn_smiles,
        results_folder=env.results_folder,
        n_conformers=getattr(al_cfg, 'n_conformers', 32),
        num_samples=getattr(al_cfg, 'num_samples', 3),
    )

    st = time.time()
    guesser.run_inference()
    print(f"RTSP inference completed in {time.time() - st:.2f}s")


if __name__ == '__main__':
    import motsart.learning.conf
    store(
        rtsp_guesser_task,
        name="rtsp_guesser_root",
        hydra_defaults=[
            "_self_",
            {"env": "test"},
            {"multihead_module": "mhfm_default"},
            {"al_cfg": "test"},
        ]
    )
    store.add_to_hydra_store()
    zen(rtsp_guesser_task).hydra_main(
        config_name="rtsp_guesser_root",
        version_base="1.3"
    )
