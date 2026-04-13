"""Learning inference: refine TS guesses with a trained GoFlow model.

Loads a GoFlow checkpoint, runs ODE integration on TS guess conformers,
and saves the refined geometries for downstream validation.
"""

import time
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from hydra_zen import zen, store

from ase.io import read, write
from ase import Atoms

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from goflow.flow_matching.multihead_flow_module import MultiHeadFlowModule

from goflow.preprocessing import process_reaction_data
from motsart.conf_default import EnvironmentConfig, ALConfig
from motsart.complex_finder.utils import PathHandler, get_rxn_data


class ALBaseInference(ABC):
    """
    A PyTorch based inference_module is required.
    """
    def __init__(self, inference_module: nn.Module, rxn_id: str, rxn_smiles: str, ts_method: str, results_folder: str = 'results'):
        self.rxn_data = get_rxn_data(rxn_id, rxn_smiles, r_or_p='r') # 'r' because TS-guess xyzs are saved in order of reactant atom-idx
        self.path_handler_ts_guess = PathHandler(rxn_id, 'r', ts_method=ts_method, results_folder=results_folder)
        self.path_handler_al = PathHandler(rxn_id, 'r', ts_method='learning', results_folder=results_folder)
        self.path_handler_al.ts_to_validate.mkdir(parents=True, exist_ok=True)

        self.load_and_set_inference_module(inference_module)
    
    @abstractmethod
    def load_and_set_inference_module(self, inference_module: nn.Module):
        """
        Loads the inference module.
        """
        raise NotImplementedError('Must be implemented.')

    
    @abstractmethod
    def infer_optimized_ts(self, ts_guesses_C_N_3: np.ndarray) -> np.ndarray:
        """
        Given multiple ts guess conformers, sorted by atom mapping, return the optimized ts for each of the conformers.
        """
        raise NotImplementedError('Must be implemented.')
    
    def run_inference(self):
        print(f"Running AL inference on rxn {self.rxn_data.rxn_id}")

        ts_conformer_xyz_files_L = self.path_handler_ts_guess.get_ts_guess_files_to_validate()
        
        ts_confs_C_N_3 = np.array([read(file).get_positions() for file in ts_conformer_xyz_files_L])

        with torch.inference_mode():
            ts_confs_opt_C_N_3 = self.infer_optimized_ts(ts_confs_C_N_3)

        assert ts_confs_C_N_3.shape == ts_confs_opt_C_N_3.shape

        for i in range(len(ts_confs_C_N_3)):
            ase_atoms = Atoms(symbols=self.rxn_data.atoms_mn_N, positions=ts_confs_opt_C_N_3[i])
            file_path = self.path_handler_al.ts_to_validate / ts_conformer_xyz_files_L[i].name
            write(file_path, ase_atoms)
        
        print(f"Saved generated xyz files to: {self.path_handler_al.ts_to_validate}")


class ALPassThroughInference(ALBaseInference):
    def infer_optimized_ts(self, ts_guesses_C_N_3: np.ndarray) -> np.ndarray :        
        return ts_guesses_C_N_3


class ALGoFlowInference(ALBaseInference):
    def load_and_set_inference_module(self, inference_module: MultiHeadFlowModule):
        with open(self.path_handler_al.learning_feat_dict, 'rb') as f:
            self.feat_dict = pickle.load(f)
        
        ckpt = torch.load(self.path_handler_al.learning_ckpt, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt # Handle Lightning checkpoint structure
        
        inference_module.load_state_dict(state_dict)
        inference_module.eval()
        self.inference_module = inference_module
        
    
    def infer_optimized_ts(self, ts_guesses_C_N_3: np.ndarray):        
        data_list_C = process_reaction_data(
            feat_dict=self.feat_dict,
            rxn_smiles=self.rxn_data.rxn_smiles,
            rxn_id=-1,
            guess_xyzs_C_N_3=ts_guesses_C_N_3,
            gt_xyzs_C_N_3=None,
        )

        self.inference_module.test_results = {head: [] for head in self.inference_module.active_heads}
        self.inference_module.test_step(Batch.from_data_list(data_list_C), 0)

        return np.stack([data.pos_gen_TS.numpy() for data in self.inference_module.test_results['TS']])


def learning_inference_task(flow_module: MultiHeadFlowModule, env: EnvironmentConfig, al_cfg: ALConfig):
    df_smi = pd.read_csv(env.rxn_csv, sep=',', header=None)
    rxn_id = df_smi[0].values[env.rxn_num]
    rxn_smiles = df_smi[1].values[env.rxn_num]

    inference_machine = ALGoFlowInference(flow_module, rxn_id, rxn_smiles, al_cfg.learning_path_guesser, results_folder=env.results_folder)
    st = time.time()
    inference_machine.run_inference()
    print(f"Inference completed in {time.time() - st:.2f}s")


if __name__ == '__main__':
    import motsart.learning.conf
    store(
        learning_inference_task,
        name="learning_inference_root",
        hydra_defaults=[
            "_self_",
            {"env": "test"},
            {"flow_module": "mhfm_default"},
            {"al_cfg": "test"},
        ]
    )
    store.add_to_hydra_store()
    zen(learning_inference_task).hydra_main(
        config_name="learning_inference_root",
        version_base="1.3"
    )

