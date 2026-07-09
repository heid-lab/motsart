"""RacerTS path guesser: refine TS guesses via conformer sampling.

Uses the RacerTS conformer generator to sample TS-like conformers around
existing RMSD++ TS guesses and selects the lowest-energy candidate after
xTB relaxation.
"""

from racerts import ConformerGenerator
from motsart.complex_finder.utils import get_rxn_data, get_xtb_energies_of_population, write_xyz, relax_pop_with_constraint
from motsart.common import PathHandler
from motsart.conf_default import EnvironmentConfig
from rdkit import Chem
import numpy as np
import pandas as pd
from hydra_zen import zen, store
import shutil
import os


def get_sorted_conformer_coords(mol: Chem.Mol) -> np.ndarray:
    """Return conformer coordinates sorted by energy, shape (C, N, 3)."""
    # 1. Get all conformers
    confs = list(mol.GetConformers())
    
    if not confs:
        return np.empty((0, mol.GetNumAtoms(), 3))

    # 2. Sort the list of conformer objects based on the property
    confs.sort(key=lambda c: c.GetDoubleProp('energy'))

    # 3. Extract positions using GetPositions() which returns (N, 3) numpy arrays
    coords_list = [c.GetPositions() for c in confs]

    # 4. Stack into a single (C, N, 3) array
    return np.array(coords_list)


def ts_sampling_and_xtb_energy_eval(rxn_data, mol_file, path_handler: PathHandler) -> np.ndarray | None:
    reacting_atoms = [rxn_data.mn_order[mn] for core_A in rxn_data.rxn_core_mn_R_A for mn in core_A]

    path_handler.chrdir_to_ts_temp()
    try:
        cg = ConformerGenerator(randomSeed=42)
        ts_conformers_mol = cg.generate_conformers(
            file_name=str(mol_file), 
            charge=rxn_data.charge,
            reacting_atoms=reacting_atoms,
            conf_factor=40,
        )
        # Sort by xTB energies
        mol_C_N_3 = get_sorted_conformer_coords(ts_conformers_mol)
        xtb_energies_C = get_xtb_energies_of_population(mol_C_N_3, rxn_data)
        sort_idx = np.argsort(xtb_energies_C)
        
        # xTB relax population
        mols_best_B_N_3 = mol_C_N_3[sort_idx[:8]]
        mols_best_B_N_3, energies_B = relax_pop_with_constraint(rxn_data, mols_best_B_N_3, path_handler, iter_n=None, use_mol_dist=True, force_const=.1)
        if len(energies_B) == 0:
            print(f"Relaxing population only returned formed bonds. Skipping rxn {rxn_data.rxn_id}, file: {mol_file}")
            return None
    except Exception as e:
        print(f"Error in racerTS {e}, skipping rxn {rxn_data.rxn_id}, file: {mol_file}")
        return None
    
    path_handler.chdir_to_proj_root()
    best_idx = np.argmin(energies_B)
    print(np.sort(energies_B))
    
    return mols_best_B_N_3[best_idx]


def ts_conf_sampler_task(env: EnvironmentConfig):
    df_smi = pd.read_csv(env.rxn_csv, sep=',', header=None)

    if env.rxn_id is not None:
        rxn_id = env.rxn_id
        matching_rows = df_smi[df_smi[0] == rxn_id]
        rxn_smiles = matching_rows[1].values[0]
    else:
        rxn_id = df_smi[0].values[env.rxn_num]
        rxn_smiles = df_smi[1].values[env.rxn_num]
    
    print(f"Starting racerTS for rxn {rxn_id}")
    rxn_data = get_rxn_data(rxn_id, rxn_smiles, env.solvent)

    path_handler = PathHandler(rxn_id, 'r', ts_method='rmsd_pp', results_folder=env.results_folder)
    path_handler_racerTS = PathHandler(rxn_id, 'r', ts_method='racer_ts', results_folder=env.results_folder)
    path_handler_racerTS.create_ts_method_dirs()

    for ts_guess_file in path_handler.ts_to_validate.glob('*.xyz'):
        racer_ts_filepath = path_handler_racerTS.ts_to_validate / ts_guess_file.name
        print(f"racerTS: writing to path {racer_ts_filepath}")
        
        better_ts_N_3 = ts_sampling_and_xtb_energy_eval(rxn_data, ts_guess_file, path_handler_racerTS)
        if better_ts_N_3 is None:
            shutil.copy(ts_guess_file, racer_ts_filepath)
        else:
            write_xyz(rxn_data.atoms_mn_N, better_ts_N_3, racer_ts_filepath)


if __name__ == '__main__':
    store(
        ts_conf_sampler_task,
        name="ts_conf_sampler_root",
        hydra_defaults=[
            "_self_",
            {"env": "test"},
        ]
    )
    store.add_to_hydra_store()
    zen(ts_conf_sampler_task).hydra_main(
        config_name="ts_conf_sampler_root",
        version_base="1.3"
    )
