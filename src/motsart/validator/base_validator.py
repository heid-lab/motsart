"""Abstract base class for TS validation.

Provides the ``validate()`` workflow that iterates over TS guess files,
delegates saddle-point optimization + IRC to subclasses, and saves results.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
from ase import Atoms
from ase.io import write
from ase.io import iread, read
from hydra_zen import zen, store

from .utils import get_adj_matrix_from_mol
from motsart.common import PathHandler
from motsart.conf_default import EnvironmentConfig, ValidatorConfig
from motsart.complex_finder.utils import (
    get_rxn_data,
    get_rdkit_mol_from_xyz
)


class BaseValidator(ABC):
    """Base class for TS validators.

    Subclasses must implement :meth:`validate_single_ts` which performs
    saddle-point optimization and IRC for a single TS guess file.

    Args:
        rxn_id: Reaction identifier.
        rxn_smiles: Reaction SMILES string.
        cfg: Validator configuration (nprocs, maxiter, etc.).
        env: Environment configuration (paths, solvent, etc.).
    """

    def __init__(self, rxn_id: str, rxn_smiles: str, cfg: ValidatorConfig, env: EnvironmentConfig):
        self.rxn_data = get_rxn_data(
            rxn_id=rxn_id,
            rxn_smiles=rxn_smiles,
            solvent=env.solvent,
            r_or_p='r'
        )
        self.cfg = cfg
        self.env = env


    @abstractmethod
    def validate_single_ts(self, ts_guess_file: Path) -> List[Dict]:
        """
        Args:
           ts_guess_file: xyz file of the TS that should be validated
        
        Returns:
            ts_results: Results of saddle-point optimization. Dict with keys.
                {'converged': bool, 'opt_xyz_N_3': np.ndarray, 'neg_imag_cnt': int, 'cycle_cnt': int, 'neg_imag_cnt': int}
                => opt_xyz_N_3 needs to be provided if converged (optimized geometry, Nx3 array)
            irc_results: Results of IRC validation. Bool indicating of IRC converged to correct reactants and products.
        """
        raise NotImplementedError("This must be implemented!")
    
    def save_results(self, results_dict: Dict, opt_xyz_N_3: Optional[np.ndarray]):
        # Save stats to CSV
        results_df = pd.DataFrame([results_dict])
        if self.path_handler.validation_results.exists():
            results_df.to_csv(self.path_handler.validation_results, mode='a', header=False, index=False)
        else:
            results_df.to_csv(self.path_handler.validation_results, mode='w', header=True, index=False)
        
        if opt_xyz_N_3 is not None:
            # Save optimized xyz (saddle-point) to xyz file
            atoms = Atoms(symbols=self.rxn_data.atoms_mn_N, positions=opt_xyz_N_3)
            output_file = self.path_handler.ts_sp_opt_gt / results_dict['ts_file']
            write(output_file, atoms)

    
    def get_R_and_P_rdkit_mols_from_IRC(self, irc_file: Path):
        """Return (reactant_atoms, product_atoms) from first/last frames of trajectory."""
        traj_T = [m for m in iread(irc_file)]
        if not traj_T or len(traj_T) < 2:
            return None, None

        # We don't know which one of them is r/p
        rdkit_mol_rp1 = get_rdkit_mol_from_xyz(traj_T[0].get_positions(), self.rxn_data.atoms_mn_N, charge=self.rxn_data.charge)
        rdkit_mol_rp2 = get_rdkit_mol_from_xyz(traj_T[-1].get_positions(), self.rxn_data.atoms_mn_N, charge=self.rxn_data.charge)

        return rdkit_mol_rp1, rdkit_mol_rp2
    

    def parse_irc_results(self, irc_traj_file: Path) -> bool:
        """
        Parse IRC validation results.
                
        Returns:
            bool (IRC converged to reactant and product)
        """
        irc_mol_rp1, irc_mol_rp2 = self.get_R_and_P_rdkit_mols_from_IRC(irc_traj_file)
        if irc_mol_rp1 is None or irc_mol_rp2 is None:
            return False

        adj_irc_1 = get_adj_matrix_from_mol(irc_mol_rp1)
        adj_irc_2 = get_adj_matrix_from_mol(irc_mol_rp2)
        if adj_irc_1 is None or adj_irc_2 is None:
            return False

        r_sm_adj = get_adj_matrix_from_mol(self.rxn_data.r_mol, self.rxn_data.r_idx_to_mn)
        p_sm_adj = get_adj_matrix_from_mol(self.rxn_data.p_mol, self.rxn_data.p_idx_to_mn)

        valid_path_1 = np.array_equal(adj_irc_1, r_sm_adj) and np.array_equal(adj_irc_2, p_sm_adj)
        valid_path_2 = np.array_equal(adj_irc_1, p_sm_adj) and np.array_equal(adj_irc_2, r_sm_adj)

        return valid_path_1 or valid_path_2

    
    def validate(self):
        """
        Main workflow for validating all TS guesses.
        For each path guessing method, there are multiple TS guesses to validate.
        This calls validate_single_ts(), which has to be implemented, for each of those files and saves results.        
        """
        for alg in self.cfg.path_guessers_to_validate:
            self.path_handler = PathHandler(self.rxn_data.rxn_id, 'r', ts_method=alg, validation_method=self.__class__.__name__, results_folder=self.env.results_folder)
            self.path_handler.rm_existing_validation_dirs()
            self.path_handler.create_validation_dirs()
            
            for ts_guess_file in self.path_handler.get_ts_guess_files_to_validate():
                print(f"\nRunning TS optimization for {alg}, {ts_guess_file} ...")

                ts_results, irc_results = self.validate_single_ts(ts_guess_file)
                results_dict = {
                    'rxn_id': self.rxn_data.rxn_id,
                    'ts_file': ts_guess_file.name,
                    'converged': ts_results['converged'],
                    'neg_imag_cnt': ts_results['neg_imag_cnt'],
                    'energy_kcal': ts_results['energy_kcal'],
                    'cycle_cnt': ts_results.get('cycle_cnt', -1),
                    'irc_converged': irc_results,
                }
                
                self.save_results(results_dict, ts_results.get('opt_xyz_N_3', None))
                
                print(f"Completed validation for {ts_guess_file.name}")
                print(f"Results: {results_dict}")


def validate_task(validator: callable, validator_cfg: ValidatorConfig, env: EnvironmentConfig):    
    df_smi = pd.read_csv(env.rxn_csv, sep=',', header=None)
    if env.rxn_id is not None:
        rxn_id = env.rxn_id
        matching_rows = df_smi[df_smi[0] == rxn_id]
        rxn_smiles = matching_rows[1].values[0]
    else:
        rxn_id = df_smi[0].values[env.rxn_num]
        rxn_smiles = df_smi[1].values[env.rxn_num]

    validator: BaseValidator = validator(
        rxn_id=rxn_id,
        rxn_smiles=rxn_smiles,
        cfg=validator_cfg,
        env=env,
    )
    
    print(f"Running {type(validator).__name__} for rxn {rxn_id}")
    validator.validate()


if __name__ == "__main__":
    import motsart.validator.conf
    store(
        validate_task,
        name="validator_root",
        hydra_defaults=[
            "_self_",
            {"env": "test"},
            {"validator_cfg": "test"},
            {"validator": "xtb"},
        ]
    )
    store.add_to_hydra_store()
    zen(validate_task).hydra_main(
        config_name="validator_root",
        version_base="1.3"
    )
