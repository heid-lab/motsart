"""
Preprocessing script for multi-head flow matching (R, TS, P prediction).

Creates PyG data objects from MOTSART results containing:
- pos_R: Reactant complex geometry
- pos_TS: Transition state geometry
- pos_P: Product complex geometry
- confs_R_N_M_3: RDKit-generated conformers for R (optional)
- confs_P_N_M_3: RDKit-generated conformers for P (optional)
- mol_name: Filename (e.g., mol_000.xyz)
- rxn_id: Reaction index

Usage:
    python -m motsart.learning.results_to_data_pkl_pre --help
"""

import random
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from ase.io import read
import numpy as np
import torch

from motsart.common import PathHandler
from goflow.preprocessing import process_reaction_data
from goflow.flow_matching.utils import generate_smiles_conformers


# ===================== Default Configuration =====================
DEFAULT_RESULTS_FOLDER = 'results_musica/results_musica'
DEFAULT_RXN_CSV = 'data/cyclo32_small_rand.csv'
DEFAULT_OUT_DIR = '/Users/leo/Documents/motsart_project/goflow/data/CYCLO_PRE/processed_data'
DEFAULT_TS_METHOD = 'racer_ts'
DEFAULT_N_CONFORMERS = 32
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1

# ===================== Helper Functions =====================

def get_pos_from_xyz(xyz_path: Path) -> np.ndarray:
    """Read xyz file and return positions as numpy array."""
    atoms = read(xyz_path)
    return atoms.get_positions()


def get_r_p_ts_triplets(path_handler: PathHandler, ts_method: str = 'racer_ts'):
    """
    Find all matching (R, P, TS) triplets across all reactions.

    Returns list of tuples: (r_file, p_file, ts_file, rxn_id, rxn_smiles, mol_name)
    """
    triplets = []

    # Pattern to find all TS files
    ts_pattern = f'R*/{path_handler.ts_foldername}/{ts_method}/{path_handler.ts_guess_foldername}/*.xyz'

    for ts_file in path_handler.results_dir.glob(ts_pattern):
        mol_name = ts_file.name

        # Parse reaction folder
        parts = ts_file.relative_to(path_handler.results_dir).parts
        rxn_folder = parts[0]  # e.g., "R0"

        # Construct matching R and P paths
        rxn_dir = path_handler.results_dir / rxn_folder
        r_file = rxn_dir / 'r' / 'final_complexes' / mol_name
        p_file = rxn_dir / 'p' / mol_name

        # Check if all three files exist
        if r_file.exists() and p_file.exists():
            rxn_id, rxn_smiles = path_handler.get_rxn_id_and_smiles_given_mol_filepath(ts_file)
            triplets.append((r_file, p_file, ts_file, rxn_id, rxn_smiles, mol_name))

    return triplets


def main(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Configuration:")
    print(f"  results_folder: {args.results_folder}")
    print(f"  rxn_csv: {args.rxn_csv}")
    print(f"  out_dir: {args.out_dir}")
    print(f"  ts_method: {args.ts_method}")
    print(f"  generate_conformers: {args.generate_conformers}")
    print(f"  n_conformers: {args.n_conformers}")
    print(f"  group_by_rxn: {args.group_by_rxn}")
    print(f"  seed: {args.seed}")
    print()

    # Initialize PathHandler
    path_handler = PathHandler(
        rxn_csv=Path(args.rxn_csv),
        results_folder=args.results_folder
    )

    # Load feature dictionary
    with open(path_handler.learning_feat_dict, 'rb') as f:
        feat_dict = pickle.load(f)

    # Get all R-P-TS triplets
    triplets = get_r_p_ts_triplets(path_handler, ts_method=args.ts_method)
    random.shuffle(triplets)

    print(f"Found {len(triplets)} R-P-TS triplets")

    # Get unique reaction IDs for splitting
    unique_rxn_ids = list({rxn_id for _, _, _, rxn_id, _, _ in triplets})
    random.shuffle(unique_rxn_ids)
    print(f"Num unique reactions: {len(unique_rxn_ids)}")

    # Compute splits
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    r_train = args.train_ratio / total_ratio
    r_val = args.val_ratio / total_ratio

    if args.group_by_rxn:
        n_train = int(r_train * len(unique_rxn_ids))
        n_val = int(r_val * len(unique_rxn_ids))

        train_rxn_ids = set(unique_rxn_ids[:n_train])
        val_rxn_ids = set(unique_rxn_ids[n_train:n_train + n_val])
        test_rxn_ids = set(unique_rxn_ids[n_train + n_val:])

    train_data_L, val_data_L, test_data_L = [], [], []
    skipped_count = 0

    for r_file, p_file, ts_file, rxn_id, rxn_smiles, mol_name in tqdm(triplets, desc="Processing triplets"):
        try:
            # Read positions from xyz files
            pos_R = get_pos_from_xyz(r_file)
            pos_P = get_pos_from_xyz(p_file)
            pos_TS = get_pos_from_xyz(ts_file)

            # Create PyG data object using process_reaction_data
            # Use TS as the "ground truth" for graph construction
            data_obj = process_reaction_data(
                feat_dict=feat_dict,
                rxn_smiles=rxn_smiles,
                rxn_id=rxn_id,
                gt_xyzs_C_N_3=[pos_TS]
            )[0]

            # Store ground truth geometries
            data_obj.pos_R = torch.tensor(pos_R).float()
            data_obj.pos_TS = torch.tensor(pos_TS).float()
            data_obj.pos_P = torch.tensor(pos_P).float()

            # Keep pos as TS for backward compatibility
            data_obj.pos = data_obj.pos_TS.clone()

            # Store metadata
            data_obj.mol_name = mol_name
            data_obj.rxn_index = rxn_id

            # Generate conformers if enabled
            if args.generate_conformers:
                try:
                    # Get R and P SMILES from reaction SMILES
                    r_smi, p_smi = rxn_smiles.split(">>")

                    # Generate conformers
                    r_confs = generate_smiles_conformers(r_smi, args.n_conformers)
                    p_confs = generate_smiles_conformers(p_smi, args.n_conformers)

                    if r_confs is not None and p_confs is not None:
                        # Shape from generate_smiles_conformers: (M_conformers, N_atoms, 3)
                        # Transpose to (N_atoms, M_conformers, 3)
                        data_obj.confs_R_N_M_3 = torch.tensor(r_confs).float().transpose(0, 1).contiguous()
                        data_obj.confs_P_N_M_3 = torch.tensor(p_confs).float().transpose(0, 1).contiguous()
                    else:
                        print(f"Warning: Failed to generate conformers for {mol_name}")
                        continue
                except Exception as e:
                    print(f"Warning: Conformer generation failed for {mol_name}: {e}")
                    continue

            # Assign to split
            if args.group_by_rxn:
                if rxn_id in train_rxn_ids:
                    train_data_L.append(data_obj)
                elif rxn_id in val_rxn_ids:
                    val_data_L.append(data_obj)
                else:
                    test_data_L.append(data_obj)
            else:
                # Will be split later
                train_data_L.append(data_obj)

        except Exception as e:
            print(f"Error processing {mol_name}: {e}")
            skipped_count += 1
            continue

    # Handle non-grouped splitting
    if not args.group_by_rxn:
        all_data = train_data_L
        random.shuffle(all_data)

        split_train = int(r_train * len(all_data))
        split_val = int((r_train + r_val) * len(all_data))

        train_data_L = all_data[:split_train]
        val_data_L = all_data[split_train:split_val]
        test_data_L = all_data[split_val:]

    print(f"\nProcessing complete:")
    print(f"  Train size: {len(train_data_L)}")
    print(f"  Val size: {len(val_data_L)}")
    print(f"  Test size: {len(test_data_L)}")
    print(f"  Skipped: {skipped_count}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save pickle files
    with open(out_dir / 'data_train.pkl', 'wb') as f:
        pickle.dump(train_data_L, f)
    with open(out_dir / 'data_val.pkl', 'wb') as f:
        pickle.dump(val_data_L, f)
    with open(out_dir / 'data_test.pkl', 'wb') as f:
        pickle.dump(test_data_L, f)

    print(f"\nSaved to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess MOTSART results into PyG data objects for multi-head flow matching."
    )
    parser.add_argument(
        "--results_folder", type=str, default=DEFAULT_RESULTS_FOLDER,
        help=f"Path to results folder relative to project root (default: {DEFAULT_RESULTS_FOLDER})"
    )
    parser.add_argument(
        "--rxn_csv", type=str, default=DEFAULT_RXN_CSV,
        help=f"Path to reaction CSV file relative to project root (default: {DEFAULT_RXN_CSV})"
    )
    parser.add_argument(
        "--out_dir", type=str, default=DEFAULT_OUT_DIR,
        help=f"Output directory for pickle files (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--ts_method", type=str, default=DEFAULT_TS_METHOD,
        help=f"TS method folder name (default: {DEFAULT_TS_METHOD})"
    )
    parser.add_argument(
        "--generate_conformers", action="store_true", default=True,
        help="Generate RDKit conformers for R and P (default: True)"
    )
    parser.add_argument(
        "--no_conformers", action="store_true",
        help="Disable conformer generation"
    )
    parser.add_argument(
        "--n_conformers", type=int, default=DEFAULT_N_CONFORMERS,
        help=f"Number of conformers to generate (default: {DEFAULT_N_CONFORMERS})"
    )
    parser.add_argument(
        "--group_by_rxn", action="store_true", default=True,
        help="Group samples by reaction ID for train/val/test split (default: True)"
    )
    parser.add_argument(
        "--no_group_by_rxn", action="store_true",
        help="Disable grouping by reaction ID (random sample split)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO,
        help=f"Train split ratio (default: {DEFAULT_TRAIN_RATIO})"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=DEFAULT_VAL_RATIO,
        help=f"Validation split ratio (default: {DEFAULT_VAL_RATIO})"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=DEFAULT_TEST_RATIO,
        help=f"Test split ratio (default: {DEFAULT_TEST_RATIO})"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )

    args = parser.parse_args()

    # Handle negation flags
    if args.no_conformers:
        args.generate_conformers = False
    if args.no_group_by_rxn:
        args.group_by_rxn = False

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
