"""RMSD-PP path guesser using xTB's built-in path search.

Runs ``xtb --path`` between reactant and product geometries to find
a minimum-energy path and extracts the highest-energy structure as
the TS guess.
"""

from typing import Dict, List
from pathlib import Path
import numpy as np
import subprocess
import shutil
import tempfile
import pandas as pd
import re
from hydra_zen import zen, store

from motsart.path_guessers.base_reaction_path_guesser import BaseReactionPathGuesser
from motsart.complex_finder.utils import get_rxn_data
from motsart.conf_default import EnvironmentConfig


class RmsdPpReactionPathGuesser(BaseReactionPathGuesser):
    def guess_reaction_path(self, reactive_complex_file: str, respective_product_file: str) -> Dict:
        """
        Run XTB path search between reactant and product geometries.
        
        Args:
            reactive_complex_file: xyz file to the reactant complex
            respective_product_file: xyz file to the respective product
            
        Returns:
            Dictionary containing:
                'ts_energy': energy of the TS
                'ts_atoms_N': list of atom symbols for the TS
                'ts_coords_N_3': TS guess coordinates (Nx3-dimensional)
                'path_I_N_3': full path from reactant complex to product (IxNx3-dimensional)
        """        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            try:
                cmd = [
                    'xtb', str(reactive_complex_file), '--path', str(respective_product_file),
                    '--chrg', str(self.rxn_data.charge),
                    '--gfn', '2'
                ]
                if self.rxn_data.solvent is not None:
                    cmd.extend(['--alpb', self.rxn_data.solvent])
                
                result = subprocess.run(
                    cmd,
                    cwd=tmp_path,
                    capture_output=True,
                    text=True,
                    timeout=720
                )
                
                if result.returncode != 0:
                    print(f"XTB path search failed: {result.stderr}")
                    return None
                
                # Parse TS file
                ts_file = tmp_path / 'xtbpath_ts.xyz'
                if not ts_file.exists():
                    print("TS geometry file not found after path search")
                    # Save output for debugging
                    log_file = self.path_handler.ts_to_validate.parent / 'xtb_output.log'
                    with open(log_file, 'w') as f:
                        f.write(result.stdout)
                    return None
                
                # Extract TS data
                ts_atoms, ts_coords = self._parse_xyz_file(ts_file)

                # Find the final path file (highest numbered)
                path_files = sorted([f for f in tmp_path.glob('xtbpath_[0-9]*.xyz')])
                rxn_path_file = self.path_handler.ts_paths / reactive_complex_file.name
                if path_files:
                    final_path_file = path_files[-1]
                    shutil.copy(final_path_file, rxn_path_file)

                path_energies_I = self.extract_energies_from_xyz(rxn_path_file)
                if len(path_energies_I) == 0:
                    print(f"No energies extracted from path file {rxn_path_file}")
                    return None
                return {
                    'path_energies_I': path_energies_I,
                    'ts_idx': np.argmax(path_energies_I),
                    'ts_atoms_N': ts_atoms,
                    'ts_coords_N_3': ts_coords,
                }
            
            except subprocess.TimeoutExpired:
                print(f"XTB path search timed out for {reactive_complex_file}. Skipping this molecule.")
                return None
            except Exception as e:
                print(f"Error during XTB path search: {e}. Skipping this molecule.")
                return None


    def extract_energies_from_xyz(self, rxn_path_file: Path) -> np.ndarray:
        """
        Extract all energies from a multi-structure XYZ file.
        
        Parameters
        ----------
        rxn_path_file : str
            Path to the XYZ file containing multiple structures
        
        Returns
        -------
        energies_I: List[int]
            List of energies extracted from each structure
        """
        energies_I = []
        
        with open(rxn_path_file, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                
                comment = f.readline()
                if not comment:
                    break
                
                # Extract energy from comment line
                energy_match = re.search(r'energy:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', comment)
                if energy_match:
                    energies_I.append(float(energy_match.group(1)))
                
                # Skip coordinate lines
                n_atoms = int(line.strip())
                for _ in range(n_atoms):
                    f.readline()
        
        return np.array(energies_I) / 627.509474  # Convert kcal/mol to Hartree

    
    def _parse_xyz_file(self, xyz_file: Path) -> tuple[list[str], np.ndarray]:
        """
        Parse XYZ file to extract atom symbols and coordinates.
        
        Returns:
            Tuple of (atom_symbols, coordinates)
        """
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        
        # Skip first two lines (number of atoms and comment)
        data_lines = [line.strip() for line in lines[2:] if line.strip()]
        
        atoms = []
        coords = []
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        return atoms, np.array(coords)


def rmsd_pp_task(env: EnvironmentConfig):
    df_smi = pd.read_csv(env.rxn_csv, sep=',', header=None)
    if env.rxn_id is not None:
        rxn_id = env.rxn_id
        matching_rows = df_smi[df_smi[0] == rxn_id]
        rxn_smiles = matching_rows[1].values[0]
    else:
        rxn_id = df_smi[0].values[env.rxn_num]
        rxn_smiles = df_smi[1].values[env.rxn_num]
    
    rxn_data = get_rxn_data(rxn_id, rxn_smiles, solvent=env.solvent, r_or_p='r')
    path_guesser = RmsdPpReactionPathGuesser(rxn_data, ts_method='rmsd_pp', results_folder=env.results_folder)
    path_guesser.compute_rxn_path_and_save_data()


if __name__ == '__main__':
    store(
        rmsd_pp_task,
        name="learning_training_root",
        hydra_defaults=[
            "_self_",
            {"env": "test"},
        ]
    )
    store.add_to_hydra_store()
    zen(rmsd_pp_task).hydra_main(
        config_name="learning_training_root",
        version_base="1.3"
    )

