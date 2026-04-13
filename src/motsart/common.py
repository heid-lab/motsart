"""Central path management for the moTSart pipeline.

Provides :class:`PathHandler`, which constructs and manages all file paths
and directory structures used across the pipeline stages (complex finder,
path guessers, validator, learning).
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd


def get_root_dir() -> Path:
    """Return the project root directory (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


class PathHandler():
    """Central utility for managing file paths and directory structures.

    Constructs paths for reaction-specific results, TS guesses, validation
    outputs, and learning data. Used throughout the pipeline to
    locate input/output files for each reaction.

    Args:
        rxn_id: Reaction identifier (e.g. ``"0"``). If provided, reaction-specific
            paths are initialized via :meth:`set_rxn_foldernames`.
        r_or_p: ``"r"`` for reactant or ``"p"`` for product directories.
        ts_method: Name of the TS guessing method (e.g. ``"rmsd_pp"``, ``"racer_ts"``).
        validation_method: Name of the validation method (e.g. ``"dft"``).
        rxn_csv: Path to the reaction CSV file (relative to project root).
        results_folder: Name of the results directory (default: ``"results"``).
    """
    def __init__(self, rxn_id: str = None, r_or_p: str = None, ts_method: str = None, validation_method: str = 'dft', rxn_csv: str = None, results_folder: str = 'results'):
        # ------------------------ Common ------------------------
        self.project_root_dir = get_root_dir()
        self.results_dir = self.project_root_dir / results_folder
        self.data_dir = self.project_root_dir / 'data'
        if rxn_csv is not None:
            self.rxn_csv = self.project_root_dir / rxn_csv
            self.df_smi = pd.read_csv(self.rxn_csv, sep=',', header=None)
        
        # ------------------------ Reaction Specific ------------------------
        self.ts_foldername = 'ts'
        self.ts_gt_foldername = 'ts_gt'
        self.ts_guess_foldername = 'ts_to_validate'
        self.validation_foldername = 'validation'

        if rxn_id is not None:
            self.set_rxn_foldernames(rxn_id, r_or_p, ts_method, validation_method)

        # ------------------------ Learning ------------------------
        self.learning_foldername = "learning_data"
        self.learning_dir = self.project_root_dir / self.learning_foldername
        self.learning_init_ckpt = self.learning_dir / 'goflow_pretrained.ckpt'
        self.learning_ckpt = self.learning_dir / 'goflow.ckpt'
        self.learning_feat_dict = self.learning_dir / 'feat_dict_organic.pkl'
        self.learning_cluster_script = self.project_root_dir / 'complex_and_ts_search_cluster_AL.sh'

        # --- Validation Stats ---
        self.validation_stats = self.results_dir / 'final_stats'


    def set_rxn_foldernames(self, rxn_id: str, r_or_p: str, ts_method: str, validation_method: str = 'dft') -> None:
        """Initialize all reaction-specific directory paths.

        Args:
            rxn_id: Reaction identifier.
            r_or_p: ``"r"`` for reactant or ``"p"`` for product.
            ts_method: TS guessing method name.
            validation_method: Validation method name (default: ``"dft"``).
        """
        # ------------------------ Complex finder ------------------------
        self.rxn_dir = self.results_dir / f'R{rxn_id}'
        self.rp_dir = self.rxn_dir / r_or_p
        self.rp_dir_temp = self.rp_dir / 'temp'
        self.rp_dir_struct_xyzs = self.rp_dir / 'struct_xyzs'
        self.p_dir = self.rxn_dir / 'p'
        
        self.rp_dir_final = self.rp_dir / 'final_complexes'
        self.final_energies_rewards_csv = self.rp_dir_final / 'energies_rewards.csv'
        
        # ------------------------ Path guesser ------------------------
        self.ts = self.rxn_dir / self.ts_foldername
        self.ts_method = self.ts / ts_method
        self.ts_temp = self.ts_method / 'temp'
        self.ts_paths = self.ts_method / 'paths'
        
        self.ts_to_validate = self.ts_method / self.ts_guess_foldername

        # ------------------------ Validator ------------------------
        self.validation = self.rxn_dir / self.validation_foldername
        self.validation_ts_method = self.validation / ts_method
        
        self.ts_sp_opt = self.validation_ts_method / 'ts_sp_opt'
        self.ts_sp_opt_orca = self.ts_sp_opt / f'orca_{validation_method}'
        
        self.irc = self.validation_ts_method / 'irc'
        self.irc_orca = self.irc / f'orca_{validation_method}'
        
        self.ts_sp_opt_gt = self.validation_ts_method / self.ts_gt_foldername   # use at ground-truth in learning
        
        self.validation_results = self.validation_ts_method / f'validation_{validation_method}.csv'
    
    
    def create_dirs(self) -> None:
        """Create all directories needed for complex finder and path guesser output."""
        os.makedirs(self.rp_dir, exist_ok=True)
        os.makedirs(self.rp_dir_temp, exist_ok=True)
        os.makedirs(self.rp_dir_struct_xyzs, exist_ok=True)
        os.makedirs(self.rp_dir_final, exist_ok=True)

        os.makedirs(self.ts_temp, exist_ok=True)
        os.makedirs(self.ts_paths, exist_ok=True)
        os.makedirs(self.ts_to_validate, exist_ok=True)
        
        os.makedirs(self.p_dir, exist_ok=True)
    
    def create_validation_dirs(self) -> None:
        """Create directories for validation output (saddle-point opt, IRC, stats)."""
        os.makedirs(self.ts_sp_opt_orca, exist_ok=True)
        os.makedirs(self.ts_sp_opt_gt, exist_ok=True)
        os.makedirs(self.irc_orca, exist_ok=True)
        os.makedirs(self.validation_stats, exist_ok=True)
    
    def rm_existing_validation_dirs(self):
        shutil.rmtree(self.ts_sp_opt_orca, ignore_errors=True)
        shutil.rmtree(self.irc_orca, ignore_errors=True)
        if self.validation_results.exists():
            os.remove(self.validation_results)
        
    def create_ts_method_dirs(self):
        os.makedirs(self.ts_temp, exist_ok=True)
        os.makedirs(self.ts_paths, exist_ok=True)
        os.makedirs(self.ts_to_validate, exist_ok=True)

    def rm_existing_ts_method_dirs(self):
        if self.ts_method.exists():
            shutil.rmtree(self.ts_method)
    
    def rm_existing_ts_dir(self):
        if self.ts.exists():
            shutil.rmtree(self.ts)
        self.create_dirs()
    
    def rm_existing_rp_dir(self):
        if self.rp_dir.exists():
            shutil.rmtree(self.rp_dir)
        if self.p_dir.exists():
            shutil.rmtree(self.p_dir)
        self.create_dirs()
            
    def get_iter_n_dir(self, n):
        struct_xyzs_iter_n = self.rp_dir_struct_xyzs / f'iter_{n}'
        os.makedirs(struct_xyzs_iter_n, exist_ok=True)
        return struct_xyzs_iter_n

    def get_ts_guess_files_to_validate(self) -> List[Path]:
        """Return sorted list of TS guess XYZ files awaiting validation."""
        return sorted(self.ts_to_validate.glob('*.xyz'))
    
    def chrdir_to_ts_temp(self):
        os.chdir(self.ts_temp)
    
    def chdir_to_temp(self):
        os.chdir(self.rp_dir_temp) # xtb writes files into here
    
    def chdir_to_proj_root(self):
        os.chdir(self.project_root_dir)

    def get_reactant_complexes_xyz_files(self) -> List[Path]:
        """Return sorted list of final reactant complex XYZ files."""
        sorted_files = sorted(self.rp_dir_final.glob('*.xyz'))
        return [Path(file) for file in sorted_files]
    
    def get_reactive_complex_and_respective_product_files(self) -> Tuple[List[Path], List[Path]]:
        """Return paired lists of (reactant_complex, product) XYZ files."""
        p_files_C = sorted(self.p_dir.glob('*.xyz'))
        rc_files_C = [self.rp_dir_final / p_file.name for p_file in p_files_C]
        return rc_files_C, p_files_C
    
    def was_IRC_success(self, rxn_id: str, ts_method: str, mol_name: str) -> bool:
        """Check if IRC validation succeeded for a given TS guess."""
        self.set_rxn_foldernames(rxn_id, 'r', ts_method)
        validation_csv_file_L = list(self.validation_ts_method.glob('validation_*.csv'))
        if len(validation_csv_file_L) == 0: return False
        
        df_val = pd.read_csv(validation_csv_file_L[0])
        row = df_val[df_val['ts_file'] == mol_name]
        
        return (len(row) > 0) and row['irc_converged'].values[0]
    
    def get_ts_guesses_and_respective_ts_gt(self, check_irc_success: bool = False) -> List[Tuple[Path, Path]]:
        """Return list of (ts_guess_file, ts_ground_truth_file) pairs across all reactions.

        Args:
            check_irc_success: If ``True``, only include pairs where IRC validation succeeded.

        Returns:
            List of ``(guess_path, ground_truth_path)`` tuples.
        """
        guess_gt_pairs_L_2 = []
        
        ts_guess_pattern = f'R*/{self.ts_foldername}/*/{self.ts_guess_foldername}/*.xyz'
        for ts_guess_filepath in self.results_dir.glob(ts_guess_pattern):
            # Parse the path to identify the Reaction and Method
            parts = ts_guess_filepath.relative_to(self.results_dir).parts
            rxn_id, ts_method, mol_name = parts[0], parts[2], parts[-1]

            # Construct the corresponding Ground Truth path
            gt_filepath = self.results_dir / rxn_id / self.validation_foldername / ts_method / self.ts_gt_foldername / mol_name
            if gt_filepath.exists():
                if (not check_irc_success) or (check_irc_success and self.was_IRC_success(rxn_id[1:], ts_method, mol_name)):
                    guess_gt_pairs_L_2.append((ts_guess_filepath, gt_filepath))

        return guess_gt_pairs_L_2
    
    def num_ts_gt_calculated_run(self):
        return len(self.get_ts_guesses_and_respective_ts_gt())

    def get_rxn_id_and_smiles_given_mol_filepath(self, filepath: Path) -> Tuple[str, str]:
        """Extract reaction ID and SMILES from a molecule file path in the results tree.

        Args:
            filepath: Path to a molecule XYZ file within the results directory.

        Returns:
            Tuple of ``(rxn_id, rxn_smiles)``.
        """
        parts = filepath.relative_to(self.results_dir).parts
        rxn_id = parts[0].lstrip('R')

        row = self.df_smi[self.df_smi[0] == int(rxn_id)]
        assert len(row) == 1, f"Expected 1 row for rxn_id {rxn_id}, found {len(row)}"
        rxn_smiles = row[1].values[0]

        return rxn_id, rxn_smiles
        