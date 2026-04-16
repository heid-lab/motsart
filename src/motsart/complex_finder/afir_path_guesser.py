"""
This module finds a product complex, corresponding to a given reactant/reactant complex. This is used downstream to predict reaction paths with a path_guesser.
The current method which solves this (AFIR) also generates a reaction path to arrive at the product complex.
This reaction path is also saved to be validated using the validator.
"""

import os
import numpy as np
from typing import List
from scipy.spatial import distance_matrix
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import time

from .utils import (
    ReactionData,
    standardized_rdkit_mol_from_smiles,
    get_rdkit_mol_from_xyz,
    generate_xtb_relaxed_conformers,
    write_pop_to_xyzs,
    write_xyz,
    get_bond_forming_breaking_constrains,
    xtb_optimize_with_applied_potentials,
    read_energy_coords_file,
    swap_atom_order_from_idx_to_mn,
    rdkit_mols_equal
)
from motsart.common import PathHandler
from motsart.path_guessers.utils import save_energies_plot
from motsart.complex_finder.conf import AFIRPathGuesserParams
from motsart.validator.utils import get_adj_matrix_from_mol


class AFIRReactionPathGuesser:
    def __init__(self, rxn_data: ReactionData, path_handler: PathHandler, cfg: AFIRPathGuesserParams):
        prod_mol_conf_C_N_3, _ = generate_xtb_relaxed_conformers(rxn_data.p_smiles, rxn_data.solvent, n_confs=2, seed=42)
        prod_mol_conf_C_N_3 = swap_atom_order_from_idx_to_mn(prod_mol_conf_C_N_3, rxn_data.p_mol)
        self.rxn_data = rxn_data
        self.path_handler = path_handler
        self.cfg = cfg
        self.bond_form_break_constraints = get_bond_forming_breaking_constrains(rxn_data, prod_mol_conf_C_N_3[0])
    
    def endpoint_is_product(self, atoms_N, mol_N_3, use_chirality=True, use_adj_approach=True):
        mol_relaxed = get_rdkit_mol_from_xyz(mol_N_3, atoms_N, use_chirality, charge=self.rxn_data.charge)
        if mol_relaxed is None:
            print("mol_relaxed is None")
            return False

        if use_adj_approach:
            adj_product = get_adj_matrix_from_mol(self.rxn_data.p_mol, self.rxn_data.p_idx_to_mn)
            adj_relaxed = get_adj_matrix_from_mol(mol_relaxed, idx_to_mn=None)
            return np.array_equal(adj_product, adj_relaxed)
        else:
            ref_mol = standardized_rdkit_mol_from_smiles(self.rxn_data.p_smiles, use_chirality)
            return rdkit_mols_equal(mol_relaxed, ref_mol, use_chirality)
    
    def eval_fc(self, rc_file, fc):
        try:
            valid_energies, valid_coords_I_N_3, valid_atoms_I_N, potentials = self.get_path_for_biased_optimization(rc_file, fc)
        except RuntimeError as e:
            print(f"!!!!!!!!!!!!! xTB execution error!: {e} !!!!!!!!!!!!!")
            return False, None, None, None
        
        true_energies = valid_energies - potentials
        is_endpoint_product = self.endpoint_is_product(valid_atoms_I_N[-1], valid_coords_I_N_3[-1], use_chirality=False)
        
        return is_endpoint_product, true_energies, valid_coords_I_N_3, valid_atoms_I_N
    
    def get_ts_data(self):
        """
        - Binary search the fc for all reactant complexes.
        - Return the highest energy structures
        """        
        reactive_complex_files_C = self.path_handler.get_reactant_complexes_xyz_files()
        
        rc_files_P = []
        mol_paths_P_I_N_3 = []
        path_energies_P_I = []
        valid_atoms_N = None
        
        for rc_file in reactive_complex_files_C:
            rc_filestem = rc_file.stem

            valid_path_energies_I, valid_path_I_N_3, valid_atoms_N, lowest_fc = self._process_reactive_complex(rc_file)
            if valid_path_energies_I is None:
                print(f"Skipping {rc_file}: product not reached")
                continue
            
            rc_files_P.append(rc_file)
            mol_paths_P_I_N_3.append(valid_path_I_N_3)
            path_energies_P_I.append(valid_path_energies_I)

            assert self.rxn_data.atoms_mn_N == valid_atoms_N

            # Optional for debugging purposes
            # write_pop_to_xyzs(valid_path_I_N_3, valid_atoms_N, out_path=self.path_handler.ts_paths / f'{rc_filestem}_fc_{lowest_fc:.6f}')
            # save_energies_plot(valid_path_energies_I, np.argmax(valid_path_energies_I), self.path_handler.ts_to_validate / f'{rc_filestem}_energies.png')
        
        if len(rc_files_P) == 0:
            print(f"No product found for rxn id {self.rxn_data.rxn_id}!")
            return

        ts_Pf_N_3, p_Pf_N_3, _, rc_files_Pf = self.filter_with_ts_vetting_heuristics(mol_paths_P_I_N_3, path_energies_P_I, rc_files_P)
        
        for i, rc_file in enumerate(rc_files_Pf):
            write_xyz(self.rxn_data.atoms_mn_N, ts_Pf_N_3[i], self.path_handler.ts_to_validate / rc_file.name)
            write_xyz(self.rxn_data.atoms_mn_N, p_Pf_N_3[i], self.path_handler.p_dir / rc_file.name)
    
    def filter_with_ts_vetting_heuristics(self, mol_P_I_N_3: List[np.ndarray], energies_P_I: List[np.ndarray], rc_files_P: List[Path]): # dims (P, I, N, 3), (P, I)
        """
        The dimension I can vary between different reaction paths (dimension P)
        """
        # ----- Step 1: reactant and product complex lower than ts -----
        r_energy_P = np.array([energies_I[0] for energies_I in energies_P_I])
        p_energy_P = np.array([energies_I[-1] for energies_I in energies_P_I])
        ts_energy_P = np.array([np.max(energies_I) for energies_I in energies_P_I])

        filter_P = (r_energy_P < ts_energy_P) & (p_energy_P < ts_energy_P)
        
        rc_files_P = [file for p, file in enumerate(rc_files_P) if filter_P[p]]
        energies_P_I = [energies_I for i, energies_I in enumerate(energies_P_I) if filter_P[i]]
        mol_P_I_N_3 = [mol_I_N_3 for i, mol_I_N_3 in enumerate(mol_P_I_N_3) if filter_P[i]]
        
        # ----- Step 2: filter population. Keep those with lowest TS energy -----
        ts_energy_P = np.array([np.max(energies_I) for energies_I in energies_P_I])
        
        sorted_pop_idx_Pf = np.argsort(ts_energy_P)[:self.cfg.num_ts_for_validation]

        rc_files_Pf = [rc_files_P[idx] for idx in sorted_pop_idx_Pf]
        mol_Pf_I_N_3 = [mol_P_I_N_3[idx] for idx in sorted_pop_idx_Pf]
        e_Pf_I = [energies_P_I[idx] for idx in sorted_pop_idx_Pf]

        # ----- Step 3: filter path. Take TS structure -----
        ts_idx_Pf = [np.argmax(energies_I) for energies_I in e_Pf_I]

        ts_Pf_N_3 = np.array([mol_I_N_3[ts_idx] for mol_I_N_3, ts_idx in zip(mol_Pf_I_N_3, ts_idx_Pf)])
        p_Pf_N_3 = np.array([mol_I_N_3[-1] for mol_I_N_3 in mol_Pf_I_N_3])
        e_Pf = np.array([energies_I[ts_idx] for energies_I, ts_idx in zip(e_Pf_I, ts_idx_Pf)])

        return ts_Pf_N_3, p_Pf_N_3, e_Pf, rc_files_Pf
    
    def save_energies_plot(self, valid_path_energies_I: np.ndarray, ts_guess_i: int, out_filename: Path):
        plt.figure(figsize=(8, 5))
        plt.plot(valid_path_energies_I * 627.509, 'o-')
        plt.axvline(ts_guess_i, color='r', linestyle='--', label='TS')
        plt.xlabel('Step')
        plt.ylabel('Energy (kcal/mol)')
        plt.title(f'{out_filename}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(self.path_handler.ts_to_validate / f'{out_filename}_energies.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _process_reactive_complex(self, rc_file: Path):
        """Process a single reactant complex: binary search for optimal fc and extract TS guess."""
        rc_filename = rc_file.name
        print(f"\nEvaluating {rc_filename} ------------------------------")
        
        curr_fc_upper_bound = self.cfg.fc_upper_bound
        curr_fc_lower_bound = self.cfg.fc_lower_bound

        product_was_reached = False
        valid_atoms_I_N = valid_path_energies_I = valid_path_I_N_3 = None

        for i in range(self.cfg.fc_binary_search_depth):
            curr_fc = self.cfg.fc_init_upper if i == 0 else (curr_fc_upper_bound + curr_fc_lower_bound) / 2
            print(f'curr_fc={curr_fc}')
            
            is_endpoint_product, true_energies_I, path_I_N_3, valid_atoms_I_N = self.eval_fc(rc_file, curr_fc)

            if is_endpoint_product:
                product_was_reached = True
                valid_path_I_N_3 = path_I_N_3
                valid_path_energies_I = true_energies_I
                curr_fc_upper_bound = curr_fc
                print("Endpoint is product.")
            else:
                curr_fc_lower_bound = curr_fc
                print("Endpoint not product.")

        print('')
        
        if product_was_reached:
            return valid_path_energies_I, valid_path_I_N_3, valid_atoms_I_N[0], curr_fc_upper_bound
        else:
            return None, None, None, None


    def get_path_for_biased_optimization(self, reactive_complex_xyz_file: Path, force_constant: float):
        """
        Perform biased optimization and retrieve valid path information.

        Parameters:
        - reactive_complex_xyz_file (str): Path to the reactant complex XYZ file.
        - force_constant (float): The force constant for the optimization.

        Returns:
        Tuple: Tuple containing valid energies, coordinates, atoms, and potentials.
        """
        log_file = xtb_optimize_with_applied_potentials(
            self.rxn_data, self.bond_form_break_constraints, reactive_complex_xyz_file,
            fc=force_constant, output_dir=self.path_handler.ts_temp
        )
        all_energies, all_coords, all_atoms = read_energy_coords_file(log_file)

        valid_energies, valid_coords, valid_atoms = [], [], []
        for i, coords in enumerate(all_coords):
            valid_coords.append(coords)
            valid_atoms.append(all_atoms[i])
            valid_energies.append(all_energies[i])

        potentials = self.determine_potential(valid_coords, force_constant)

        return np.array(valid_energies), np.array(valid_coords), valid_atoms, np.array(potentials)


    def determine_potential(self, all_coords: list, force_constant: float) -> List[float]:
        """
        Determine the potential energy for a set of coordinates based on distance constraints and a force constant.

        Args:
            all_coords: A list of coordinate arrays.
            force_constant: The force constant to apply to the constraints.

        Returns:
            A list of potential energy values.
        """
        potentials = []
        for coords in all_coords:
            potential = 0
            dist_matrix = distance_matrix(coords, coords)
            for a_mn_2, val in self.bond_form_break_constraints.items():
                actual_distance = dist_matrix[a_mn_2[0], a_mn_2[1]] - val
                potential += force_constant * angstrom_to_bohr(actual_distance) ** 2 # Note: original code did not have 0.5 factor, but it should be there
            potentials.append(potential)

        return potentials


def angstrom_to_bohr(distance_angstrom):
    return distance_angstrom * 1.88973


def guess_product_from_reactive_complex(rxn_data: ReactionData, path_handler: PathHandler, afir_cfg: AFIRPathGuesserParams):
    rxn_path_guesser = AFIRReactionPathGuesser(rxn_data, path_handler, afir_cfg)
    start_time = time.perf_counter()
    rxn_path_guesser.get_ts_data()
    print(f"AFIR round took {time.perf_counter() - start_time} s.")
