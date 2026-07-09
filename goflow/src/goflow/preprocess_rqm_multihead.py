"""
Preprocessing script for multi-head flow matching (R, TS, P prediction).

Key differences from preprocess_rqm.py:
- No R/P flipping - one sample per reaction
- Stores 3 ground truth geometries: pos_R, pos_TS, pos_P
- Stores 2 conformer sets: confs_R_N_M_3, confs_P_N_M_3
"""

import h5py
import pandas as pd
import pickle
import numpy as np
import torch
import os
import argparse
import random
from ase.symbols import chemical_symbols
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
from tqdm import tqdm
import io
from pathlib import Path

from .preprocessing import process_reaction_data
from goflow.flow_matching.utils import generate_smiles_conformers

RDLogger.DisableLog('rdApp.*')
ps = Chem.SmilesParserParams()
ps.removeHs = False


def save_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def get_adj_matrix_from_mol(mol: Chem.Mol, idx_to_mn=None):
    """Returns adjacency matrix from a mol, sorted by Atom Map Number."""
    mol.UpdatePropertyCache(strict=False)
    if idx_to_mn is not None:
        mol = Chem.RenumberAtoms(mol, np.argsort(idx_to_mn).tolist())
    return Chem.GetAdjacencyMatrix(mol)


def get_rdkit_mol_from_xyz(xyz_N_3: np.ndarray, atoms_N, use_chirality=False):
    """Convert xyz coordinates to RDKit molecule."""
    xyz_buf = io.StringIO()
    xyz_buf.write(f"{len(atoms_N)}\n\n")
    for atom, coord in zip(atoms_N, xyz_N_3):
        xyz_buf.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

    mol = Chem.MolFromXYZBlock(xyz_buf.getvalue())
    if mol is None:
        return None

    try:
        rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol, catchErrors=True)
        if use_chirality:
            Chem.AssignStereochemistryFrom3D(mol)
        return mol
    except Exception:
        return None


def get_adj_mat_from_smiles(smi: str):
    """Get adjacency matrix from SMILES string."""
    r_mol = Chem.MolFromSmiles(smi, params=ps)
    if r_mol is None:
        return None
    r_idx_to_mn = np.array([atom.GetAtomMapNum() for atom in r_mol.GetAtoms()])
    return get_adj_matrix_from_mol(r_mol, r_idx_to_mn)


def get_lowest_energy_irc_adj_match(r_sm_adj, rxn_dict, return_energy=False):
    """Find lowest energy IRC structure matching the adjacency matrix."""
    match_and_lowest_energy = 1e9
    match_and_lowest_energy_i = -1
    for i in range(len(rxn_dict['coords'])):
        z = rxn_dict['numbers']
        z_symbols = [Chem.GetPeriodicTable().GetElementSymbol(int(atomic_num)) for atomic_num in z]
        coords = rxn_dict['coords'][i]
        energy = rxn_dict['energies'][i]

        mol = get_rdkit_mol_from_xyz(coords, z_symbols)
        if mol is None:
            continue

        irc_adj = get_adj_matrix_from_mol(mol)
        adj_do_match = np.array_equal(irc_adj, r_sm_adj)

        if adj_do_match and energy < match_and_lowest_energy:
            match_and_lowest_energy = energy
            match_and_lowest_energy_i = i

    if match_and_lowest_energy_i == -1:
        return (None, None) if return_energy else None

    coords = rxn_dict['coords'][match_and_lowest_energy_i]
    energy = rxn_dict['energies'][match_and_lowest_energy_i]

    if return_energy:
        return coords, energy
    return coords


def get_ts_energy_from_irc(rxn_dict):
    """Get the transition state energy (maximum energy along the IRC)."""
    return float(np.max(rxn_dict['energies']))


def get_smiles_from_xyz(xyz_N_3: np.ndarray, atoms_N, use_chirality=True):
    """Convert xyz coordinates to SMILES with atom map numbers."""
    xyz_buf = io.StringIO()
    xyz_buf.write(f"{len(atoms_N)}\n\n")

    for atom, coord in zip(atoms_N, xyz_N_3):
        element_symbol = chemical_symbols[atom]
        xyz_buf.write(f"{element_symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

    mol = Chem.MolFromXYZBlock(xyz_buf.getvalue())
    if mol is None:
        return None

    try:
        rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol, catchErrors=True)

        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)

        if use_chirality:
            Chem.AssignStereochemistryFrom3D(mol)

    except Exception:
        return None

    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=use_chirality)
    return smiles


def get_sorted_atoms_from_smiles(smi: str):
    """Parse SMILES and return atomic numbers sorted by atom map number."""
    mol = Chem.MolFromSmiles(smi, params=ps)
    if mol is None:
        return None, None

    idx_to_mn = np.array([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
    atoms_sorted = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])[np.argsort(idx_to_mn)]

    return atoms_sorted, idx_to_mn


def get_concatenated_coords(data_dict):
    """Extract coordinates from H5 data dict."""
    coords_list = []
    if 'coords' in data_dict:
        return data_dict['coords'][:]

    for s_key in sorted(data_dict.keys()):
        if s_key == 'coords':
            continue
        coords_list.append(data_dict[s_key]['coords'][:])
    return np.concatenate(coords_list, axis=0)


def main(args):
    # 1. Load resources
    print(f"Loading files...")
    h5file = h5py.File(args.h5_path, 'r')
    rxn_df = pd.read_csv(args.csv_path)

    rxn_smiles_lookup = dict(zip(rxn_df['reaction_id'], rxn_df['reaction_smiles']))

    with open(args.feat_dict_path, "rb") as f:
        feat_dict_organic = pickle.load(f)

    with open(args.irc_dict, "rb") as f:
        rxn_irc_dict = pickle.load(f)

    # Single list for all data (no R/P flipping)
    pyg_data = []

    # 2. Iterate through H5 entries
    print(f"Processing reactions for multi-head setup...")
    processed_count = 0
    skipped_count = 0

    for rxn_name_key in tqdm(h5file.keys(), desc="Converting to PyG"):
        if args.max_reactions > 0 and processed_count >= args.max_reactions:
            break

        reaction_point_dict = h5file[rxn_name_key]
        rxn_smiles = rxn_smiles_lookup.get(rxn_name_key)

        if rxn_smiles is None:
            skipped_count += 1
            continue

        r_smi, p_smi = rxn_smiles.split(">>")

        # Get adjacency matrices
        r_adj = get_adj_mat_from_smiles(r_smi)
        p_adj = get_adj_mat_from_smiles(p_smi)
        if r_adj is None or p_adj is None:
            skipped_count += 1
            continue

        if rxn_name_key not in rxn_irc_dict:
            skipped_count += 1
            continue

        # Get sorted atoms
        r_atoms_N, _ = get_sorted_atoms_from_smiles(r_smi)
        p_atoms_N, _ = get_sorted_atoms_from_smiles(p_smi)
        if (r_atoms_N is None or p_atoms_N is None) or not np.array_equal(r_atoms_N, p_atoms_N):
            skipped_count += 1
            continue

        # Get R, TS, P coordinates and energies from IRC
        r_coords_mn, r_energy = get_lowest_energy_irc_adj_match(r_adj, rxn_irc_dict[rxn_name_key], return_energy=True)
        p_coords_mn, p_energy = get_lowest_energy_irc_adj_match(p_adj, rxn_irc_dict[rxn_name_key], return_energy=True)
        ts_coords_mn = get_concatenated_coords(reaction_point_dict['TS'])
        ts_energy = get_ts_energy_from_irc(rxn_irc_dict[rxn_name_key])

        if r_coords_mn is None or p_coords_mn is None:
            skipped_count += 1
            continue

        # Get SMILES from IRC coordinates
        r_smi_irc = get_smiles_from_xyz(r_coords_mn, r_atoms_N)
        p_smi_irc = get_smiles_from_xyz(p_coords_mn, p_atoms_N)

        if r_smi_irc is None or p_smi_irc is None:
            skipped_count += 1
            continue

        # Sanity check
        mol = Chem.MolFromSmiles(r_smi, params=ps)
        if mol is None:
            skipped_count += 1
            continue
        map_numbers = np.array([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
        if len(map_numbers) != ts_coords_mn.shape[0]:
            skipped_count += 1
            continue

        # Generate conformers for BOTH R and P
        try:
            r_confs_M_N_3 = generate_smiles_conformers(r_smi_irc, args.n_confs_per_rxn)
            p_confs_M_N_3 = generate_smiles_conformers(p_smi_irc, args.n_confs_per_rxn)
        except Exception as e:
            print(f"Unable to embed r/p smiles for {rxn_name_key}: {e}")
            skipped_count += 1
            continue

        # Create SINGLE data object per reaction (no flipping)
        # Use R->P direction for graph features
        data_obj = process_reaction_data(
            feat_dict=feat_dict_organic,
            rxn_smiles=f'{r_smi_irc}>>{p_smi_irc}',
            rxn_id=rxn_name_key,
            gt_xyzs_C_N_3=[ts_coords_mn]  # TS as reference for graph construction
        )[0]

        # Store ALL three ground truth geometries
        data_obj.pos_R = torch.tensor(r_coords_mn).float()
        data_obj.pos_TS = torch.tensor(ts_coords_mn).float()
        data_obj.pos_P = torch.tensor(p_coords_mn).float()

        # Store energies (in Hartree)
        data_obj.energy_r = torch.tensor(r_energy).float()
        data_obj.energy_ts = torch.tensor(ts_energy).float()
        data_obj.energy_p = torch.tensor(p_energy).float()

        # Store conformers for BOTH R and P
        # Shape: (N_atoms, M_conformers, 3)
        data_obj.confs_R_N_M_3 = torch.tensor(r_confs_M_N_3).float().transpose(0, 1).contiguous()
        data_obj.confs_P_N_M_3 = torch.tensor(p_confs_M_N_3).float().transpose(0, 1).contiguous()

        # Keep pos as TS for backward compatibility (used by some data utils)
        data_obj.pos = data_obj.pos_TS.clone()

        pyg_data.append(data_obj)
        processed_count += 1

    print(f"Successfully processed: {len(pyg_data)} reactions")
    print(f"Skipped: {skipped_count} reactions")

    # 3. Random Splitting (no doubling since no flipping)
    print(f"Splitting data...")
    random.seed(args.seed)

    rxn_indices = list(range(len(pyg_data)))
    random.shuffle(rxn_indices)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    r_train = args.train_ratio / total_ratio
    r_val = args.val_ratio / total_ratio

    n_total = len(rxn_indices)
    n_train = int(n_total * r_train)
    n_val = int(n_total * r_val)

    train_indices = rxn_indices[:n_train]
    val_indices = rxn_indices[n_train:n_train + n_val]
    test_indices = rxn_indices[n_train + n_val:]

    # Collect data for each split (no doubling)
    train_data = [pyg_data[i] for i in train_indices]
    val_data = [pyg_data[i] for i in val_indices]
    test_data = [pyg_data[i] for i in test_indices]

    # 4. Saving
    print(f"Saving files to {args.out_dir}...")
    save_pickle(train_data, os.path.join(args.out_dir, "data_train.pkl"))
    save_pickle(val_data, os.path.join(args.out_dir, "data_val.pkl"))
    save_pickle(test_data, os.path.join(args.out_dir, "data_test.pkl"))

    print(f"Done. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Data attributes per sample: pos_R, pos_TS, pos_P, energy_r, energy_ts, energy_p, confs_R_N_M_3, confs_P_N_M_3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RQM dataset for multi-head flow matching.")
    parser.add_argument("--h5_path", type=str, default="data/RQM/raw_data/B3LYPD3_TZVP.h5")
    parser.add_argument("--csv_path", type=str, default="data/RQM/raw_data/B3LYPD3_TZVP_reaction_info.csv")
    parser.add_argument("--feat_dict_path", type=str, default="data/RDB7/feat_dict_organic.pkl")
    parser.add_argument("--irc_dict", type=str, default="data/RQM/processed_data/irc_dict.pkl")
    parser.add_argument("--out_dir", type=str, default="data/RQM/processed_data/")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--n_confs_per_rxn", type=int, default=32)
    parser.add_argument("--max_reactions", type=int, default=-1, help="Max reactions to process (-1 for all)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)
