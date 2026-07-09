"""Update RQM_MH processed data with missing chi_center_C attributes.

This script recomputes chiral center indices for reactant/product
from atom-mapped SMILES and patches each Data object in-place.
"""

import argparse
import os
import pickle
from typing import List

import numpy as np
import torch
from rdkit import RDLogger

from goflow.preprocessing import get_mol, get_cip_tetra_atoms_in_decreasing_order

RDLogger.DisableLog("rdApp.*")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def compute_chi_center_indices(smiles: str):
    """Return (cip_decr_chi_nbrs_C_4, chi_center_C, rs_tag_C) for a SMILES."""
    mol = get_mol(smiles)
    if mol is None:
        raise ValueError("Unable to parse SMILES")

    perm = np.array([a.GetAtomMapNum() for a in mol.GetAtoms()]) - 1
    if (perm < 0).any():
        raise ValueError("SMILES missing atom map numbers")

    perm_inv = np.argsort(perm)
    return get_cip_tetra_atoms_in_decreasing_order(mol, perm_inv)


def update_split(pkl_path: str):
    data_list = load_pickle(pkl_path)

    updated = 0
    skipped = 0
    for data in data_list:
        try:
            if not hasattr(data, "smiles"):
                skipped += 1
                continue
            r_smi, p_smi = data.smiles.split(">>")

            _, r_chi_center_C, _ = compute_chi_center_indices(r_smi)
            _, p_chi_center_C, _ = compute_chi_center_indices(p_smi)

            data.r_chi_center_C_index = r_chi_center_C.clone()
            data.p_chi_center_C_index = p_chi_center_C.clone()
            updated += 1
        except Exception:
            skipped += 1
            continue

    save_pickle(data_list, pkl_path)
    return updated, skipped, len(data_list)


def main(args):
    split_files = [
        os.path.join(args.data_dir, "data_train.pkl"),
        os.path.join(args.data_dir, "data_val.pkl"),
        os.path.join(args.data_dir, "data_test.pkl"),
    ]

    if args.backup:
        for path in split_files:
            if os.path.exists(path):
                backup_path = f"{path}.bak"
                if not os.path.exists(backup_path):
                    with open(path, "rb") as src, open(backup_path, "wb") as dst:
                        dst.write(src.read())

    total_updated = 0
    total_skipped = 0
    total_items = 0
    for path in split_files:
        if not os.path.exists(path):
            continue
        updated, skipped, total = update_split(path)
        total_updated += updated
        total_skipped += skipped
        total_items += total
        print(f"Updated {updated}/{total} items in {os.path.basename(path)} (skipped {skipped}).")

    print(f"Done. Updated {total_updated}/{total_items}, skipped {total_skipped}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch RQM_MH pickles with chi_center_C indices.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/RQM_MH/processed_data",
        help="Directory containing data_train/val/test.pkl",
    )
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Disable .bak backups",
    )
    parser.set_defaults(backup=True)
    args = parser.parse_args()

    main(args)
