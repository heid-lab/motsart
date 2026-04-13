"""Abstract base class for reaction path guessing algorithms.

Subclasses implement specific path-finding methods (RMSD++, RacerTS, etc.)
that generate TS guess geometries from reactant complexes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
from pathlib import Path

from motsart.complex_finder.utils import ReactionData, write_pop_to_xyzs, write_xyz
from motsart.common import PathHandler
from .utils import save_energies_plot


class BaseReactionPathGuesser(ABC):
    """
    Base class for reaction path guessing algorithms.

    Provides common functionality while requiring subclasses to implement algorithm-specific path generation logic.

    Subclasses must implement:
        - guess_reaction_path(): Main method to generate TS guesses
    """

    def __init__(self, rxn_data: ReactionData, ts_method: str, results_folder: str = 'results'):
        """
        Initialize the path guesser with reaction data and path management.

        Args:
            rxn_data: Contains reaction SMILES, atom mappings, charge, solvent, etc.
            ts_method: Name of the TS guessing method (e.g. ``'rmsd_pp'``, ``'racer_ts'``).
            results_folder: Relative path to results folder from project root.
        """
        self.rxn_data = rxn_data
        self.path_handler = PathHandler(rxn_data.rxn_id, 'r', ts_method, results_folder=results_folder)
        self.path_handler.rm_existing_ts_method_dirs()
        self.path_handler.create_ts_method_dirs()
    
    @abstractmethod
    def guess_reaction_path(self, reactive_complex_file: str, respective_product_file: str) -> Dict | None:
        """
        Args:
            reactive_complex_file: xyz file to the reactant complex
            respective_product_file: xyz file to the respective product, at which one arrives when starting the reaction at the reactive_complex
        
        Returns:
            Dictionary containing:
                Required:
                'ts_atoms_N': list of atom symbols for the TS (N-dimenstional)
                'ts_coords_N_3': TS guess coordinates (Nx3-dimensional)
                
                Optional:
                'path_I_N_3': full path (I images) from reactant complex to product (IxNx3-dimensional)
                'path_energies_I: energies of the path. If provided a plot of it will be created.
                'ts_idx': required for plotting the energies.
            Returns None if path guessing fails.
        """
        raise NotImplementedError("This must be implemented!")
    
    def compute_rxn_path_and_save_data(self):
        """
        Generate transition state guesses for the reaction.

        This is the main entry point for any path guesser. It:
        1. Generates one or more reaction paths from reactants toward products
        2. Writes reaction path and ts structures to validate
         """        
        reactive_complex_files_C, respective_product_files_C = self.path_handler.get_reactive_complex_and_respective_product_files()
        if len(reactive_complex_files_C) == 0:
            print(f"No r/p files found! Skipping rxn {self.rxn_data.rxn_id}")
        
        for rc_file, p_file in zip(reactive_complex_files_C, respective_product_files_C):
            rc_filename = Path(rc_file).name
            print(f"Evaluating {rc_filename} ------------------------------")
            
            res = self.guess_reaction_path(rc_file, p_file)  
            if res is None:
                print(f"Path guessing failed. Skipping {rc_filename}.")
                continue
            
            write_xyz(res['ts_atoms_N'], res['ts_coords_N_3'], self.path_handler.ts_to_validate / rc_filename)
            #if 'path_I_N_3' in res:
            #    write_pop_to_xyzs(res['path_I_N_3'], res['ts_atoms_N'], self.path_handler.ts_paths)
            #if 'path_energies_I' in res and 'ts_idx' in res:
            #    save_energies_plot(res['path_energies_I'], res['ts_idx'],  self.path_handler.ts_to_validate / f'{Path(rc_file).stem}_energies.png')  
