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
import sys

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
    """
    Returns adjacency matrix from a mol, sorted by Atom Map Number.
    """
    mol.UpdatePropertyCache(strict=False)
    if idx_to_mn is not None:
        mol = Chem.RenumberAtoms(mol, np.argsort(idx_to_mn).tolist())
    return Chem.GetAdjacencyMatrix(mol)


def get_rdkit_mol_from_xyz(xyz_N_3: np.ndarray, atoms_N, use_chirality=False):
    # Convert np array with corresponding atom types to xyz string
    xyz_buf = io.StringIO()
    xyz_buf.write(f"{len(atoms_N)}\n\n")
    for atom, coord in zip(atoms_N, xyz_N_3):
        xyz_buf.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    
    # Get rdkit molecule from xyz string
    mol = Chem.MolFromXYZBlock(xyz_buf.getvalue())
    if mol is None:
        return None
    
    try:
        rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol, catchErrors=True)
        if use_chirality:
            Chem.AssignStereochemistryFrom3D(mol)
        return mol
    except Exception as e:
        #print(f"Error in retrieving bonds: {e}")
        return None
    
def get_adj_mat_from_smiles(smi: str):
    r_mol = Chem.MolFromSmiles(smi, params=ps)
    if r_mol is None: return None
    r_idx_to_mn = np.array([atom.GetAtomMapNum() for atom in r_mol.GetAtoms()])
    return get_adj_matrix_from_mol(r_mol, r_idx_to_mn)


def get_lowest_energy_irc_adj_match(r_sm_adj, rxn_dict, return_energy=False):
    match_and_lowest_energy = 1e9
    match_and_lowest_energy_i = -1
    for i in range(len(rxn_dict['coords'])):
        z = rxn_dict['numbers']
        z_symbols = [Chem.GetPeriodicTable().GetElementSymbol(int(atomic_num)) for atomic_num in z]
        coords = rxn_dict['coords'][i]
        energy = rxn_dict['energies'][i]

        mol = get_rdkit_mol_from_xyz(coords, z_symbols)
        if mol is None: continue

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


def create_irc_dict(args):
    out_dir = Path(args.out_dir)
    
    irc_4_15 = h5py.File(out_dir / 'B3LYPD3_TZVP_IRC_4_15.h5', 'r')
    irc_16_20 = h5py.File(out_dir / 'B3LYPD3_TZVP_IRC_16_20.h5', 'r')
    irc_21_33 = h5py.File(out_dir / 'B3LYPD3_TZVP_IRC_21_33.h5', 'r')
    
    rxn_irc_dict = {}
    for irc_h5 in [irc_4_15, irc_16_20, irc_21_33]:
        for num_atoms_key, TS_dict in irc_h5.items():
            for TS_name_key, TS_info_dict in TS_dict.items():
                rxn_irc_dict[TS_name_key] = {
                    'numbers': TS_info_dict['numbers'][:],
                    'coords': TS_info_dict['coords'][:],
                    'energies': TS_info_dict['energies'][:]
                }
    
    with open (out_dir / 'irc_dict.pkl', 'wb') as f:
        pickle.dump(rxn_irc_dict, f)


def get_smiles_from_xyz(xyz_N_3: np.ndarray, atoms_N, use_chirality=True):
    # 1. Convert np array to xyz string
    xyz_buf = io.StringIO()
    xyz_buf.write(f"{len(atoms_N)}\n\n")
    
    # Write atoms. Since input is sorted by map num, row 0 = map:1, row 1 = map:2...
    for atom, coord in zip(atoms_N, xyz_N_3):
        element_symbol = chemical_symbols[atom]
        xyz_buf.write(f"{element_symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    
    # 2. Create raw RDKit molecule
    mol = Chem.MolFromXYZBlock(xyz_buf.getvalue())
    if mol is None:
        print("Error: Could not create molecule from XYZ block")
        return None
    
    try:
        rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol, catchErrors=True)
        
        # 3. ASSIGN ATOM MAPS
        # Because input is sorted by map number, we assign i+1
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)
            
        # 4. Assign Stereochemistry
        if use_chirality:
            Chem.AssignStereochemistryFrom3D(mol)
            
    except Exception as e:
        print(f"Error in processing molecule: {e}")
        return None
    
    # 5. Generate SMILES
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=use_chirality)
    
    return smiles

def get_sorted_atoms_from_smiles(smi: str):
    """
    Parse SMILES and return atomic numbers sorted by atom map number.
    
    Args:
        smi: SMILES string with atom map numbers
        
    Returns:
        tuple: (sorted_atoms, idx_to_mn) or (None, None) if parsing fails
    """
    mol = Chem.MolFromSmiles(smi, params=ps)
    if mol is None:
        return None, None
    
    idx_to_mn = np.array([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
    atoms_sorted = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])[np.argsort(idx_to_mn)]
    
    return atoms_sorted, idx_to_mn


def main(args):
    if args.create_irc_dict:
        create_irc_dict(args)
    
    # 1. Load resources ---------------------------------------------------
    print(f"Loading files...")
    h5file = h5py.File(args.h5_path, 'r')
    rxn_df = pd.read_csv(args.csv_path)
    
    rxn_smiles_lookup = dict(zip(rxn_df['reaction_id'], rxn_df['reaction_smiles']))

    with open(args.feat_dict_path, "rb") as f:
        feat_dict_organic = pickle.load(f)

    with open(args.irc_dict, "rb") as f:
        rxn_irc_dict = pickle.load(f)
    
    r_pyg_data = []
    p_pyg_data = []
    # 2. Iterate through H5 entries ---------------------------------------------------
    print(f"Processing reactions...")
    i=0
    for rxn_name_key in tqdm(h5file.keys(), desc="Converting to PyG"):
        i+=1
        if i>4260:break
        reaction_point_dict = h5file[rxn_name_key]
        rxn_smiles = rxn_smiles_lookup.get(rxn_name_key)
        
        r_smi, p_smi = rxn_smiles.split(">>")
        if args.include_r_pos:
            r_adj = get_adj_mat_from_smiles(r_smi)
            p_adj = get_adj_mat_from_smiles(p_smi)
            if r_adj is None or p_adj is None:
                print(f"{rxn_name_key} r_mol or p_mol was none.")
                continue
            if rxn_name_key not in rxn_irc_dict:
                print(f"{rxn_name_key} not in IRC")
                continue
            
            r_atoms_N, _ = get_sorted_atoms_from_smiles(r_smi)
            p_atoms_N, _ = get_sorted_atoms_from_smiles(p_smi)
            if (r_atoms_N is None or p_atoms_N is None) or not np.array_equal(r_atoms_N, p_atoms_N):
                print(f"{rxn_name_key} could not parse r_smi or p_smi.")
                continue
    
            r_coords_mn, r_energy = get_lowest_energy_irc_adj_match(r_adj, rxn_irc_dict[rxn_name_key], return_energy=True)
            p_coords_mn, p_energy = get_lowest_energy_irc_adj_match(p_adj, rxn_irc_dict[rxn_name_key], return_energy=True)
            ts_energy = get_ts_energy_from_irc(rxn_irc_dict[rxn_name_key])

            if r_coords_mn is None or p_coords_mn is None:
                print(f"can't process rxn {rxn_name_key}. skipping")
                continue
        
            r_smi_irc = get_smiles_from_xyz(r_coords_mn, r_atoms_N)
            p_smi_irc = get_smiles_from_xyz(p_coords_mn, p_atoms_N)

        def get_concatenated_coords(data_dict):
            coords_list = []
            if 'coords' in data_dict:
                return data_dict['coords'][:]
            
            for s_key in sorted(data_dict.keys()):
                if s_key == 'coords': continue
                coords_list.append(data_dict[s_key]['coords'][:])
            return np.concatenate(coords_list, axis=0)

        ts_coords_mn = get_concatenated_coords(reaction_point_dict['TS'])

        # Sanity assert ----------------------------------------------
        mol = Chem.MolFromSmiles(r_smi, params=ps)
        if mol is None:
            print(f"Got None mol. Skipping rxn {rxn_name_key}")
            continue
        map_numbers = np.array([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
        if len(map_numbers) != ts_coords_mn.shape[0]:
            print(f"Num atoms from h5 ({len(ts_coords_mn)}) does not match smiles ({len(map_numbers)}) in rxn {rxn_name_key}")
            continue
        # Check if can be embedded. There are nonsense smiles in the dataset, e.g. [N:1](=[C:2]=[O:3])[C:5]1([H:12])[C:6]([H:13])=[P:7]#[P:8]([N:4]([H:10])[H:11])[O:9]1
        # => forces a triple bond into a 5-membered ring
        try:
            r_confs_M_N_3 = generate_smiles_conformers(r_smi_irc, args.n_confs_per_rxn)
            p_confs_M_N_3 = generate_smiles_conformers(p_smi_irc, args.n_confs_per_rxn)
        except Exception as e:
            print(f"Unable to embed r/p smiles: {e}")
            continue
        
        # Create PyG data object -------------------------------------------------
        data_obj_rp = process_reaction_data(
            feat_dict=feat_dict_organic,
            rxn_smiles=f'{r_smi_irc}>>{p_smi_irc}',
            rxn_id=f'{rxn_name_key}_r',
            gt_xyzs_C_N_3=[ts_coords_mn]
        )[0]
        data_obj_pr = process_reaction_data(
            feat_dict=feat_dict_organic,
            rxn_smiles=f'{p_smi_irc}>>{r_smi_irc}',
            rxn_id=f'{rxn_name_key}_p',
            gt_xyzs_C_N_3=[ts_coords_mn]
        )[0]

        if args.include_r_pos:
            data_obj_rp.pos = torch.tensor(r_coords_mn).float()
            data_obj_pr.pos = torch.tensor(p_coords_mn).float()
            data_obj_rp.confs_N_M_3 = torch.tensor(r_confs_M_N_3).float().transpose(0, 1).contiguous()
            data_obj_pr.confs_N_M_3 = torch.tensor(p_confs_M_N_3).float().transpose(0, 1).contiguous()
            # Store energies: R, TS, P for the forward direction (R->P)
            data_obj_rp.energy_r = torch.tensor(r_energy).float()
            data_obj_rp.energy_ts = torch.tensor(ts_energy).float()
            data_obj_rp.energy_p = torch.tensor(p_energy).float()
            # Store energies: P, TS, R for the reverse direction (P->R)
            data_obj_pr.energy_r = torch.tensor(p_energy).float()
            data_obj_pr.energy_ts = torch.tensor(ts_energy).float()
            data_obj_pr.energy_p = torch.tensor(r_energy).float()
        
        r_pyg_data.append(data_obj_rp)
        p_pyg_data.append(data_obj_pr)

    # 3. Random Splitting ---------------------------------------------------
    print(f"Total reactions: {len(r_pyg_data)}. Splitting data...")
    random.seed(args.seed)
    
    # Create indices and shuffle them to split reactions (not individual data objects)
    rxn_indices = list(range(len(r_pyg_data)))
    random.shuffle(rxn_indices)

    # Normalize ratios in case they don't sum to 1
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    r_train = args.train_ratio / total_ratio
    r_val = args.val_ratio / total_ratio

    n_total = len(rxn_indices)
    n_train = int(n_total * r_train)
    n_val = int(n_total * r_val)

    train_indices = rxn_indices[:n_train]
    val_indices = rxn_indices[n_train : n_train + n_val]
    test_indices = rxn_indices[n_train + n_val :]

    # Collect both r and p data objects for each split
    train_data = [r_pyg_data[i] for i in train_indices] + [p_pyg_data[i] for i in train_indices]
    val_data = [r_pyg_data[i] for i in val_indices] + [p_pyg_data[i] for i in val_indices]
    test_data = [r_pyg_data[i] for i in test_indices] + [p_pyg_data[i] for i in test_indices]

    # 4. Saving
    print(f"Saving files to {args.out_dir}...")
    save_pickle(train_data, os.path.join(args.out_dir, "data_train.pkl"))
    save_pickle(val_data, os.path.join(args.out_dir, "data_val.pkl"))
    save_pickle(test_data, os.path.join(args.out_dir, "data_test.pkl"))
    
    print(f"Done. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Reaction-QM dataset.")
    parser.add_argument("--h5_path", type=str, default="data/RQM/raw_data/B3LYPD3_TZVP.h5")
    parser.add_argument("--csv_path", type=str, default="data/RQM/raw_data/B3LYPD3_TZVP_reaction_info.csv")
    parser.add_argument("--feat_dict_path", type=str, default="data/RDB7/feat_dict_organic.pkl")
    parser.add_argument("--irc_dict", type=str, default="data/RQM/processed_data/irc_dict.pkl")
    parser.add_argument("--out_dir", type=str, default="data/RQM/processed/")
    parser.add_argument("--create_irc_dict", default=False, action="store_true")
    parser.add_argument("--include_r_pos", action='store_true', default=False)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--n_confs_per_rxn", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    main(args)