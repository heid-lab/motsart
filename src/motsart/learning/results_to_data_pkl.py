"""
Preprocessing script for TS guess-to-ground-truth learning data.

Creates PyG data objects from MOTSART results containing:
- guess_xyzs: TS guess geometries
- gt_xyzs: Ground-truth (validated + IRC-confirmed) TS geometries
- mol_name: Filename (e.g., mol_000.xyz)
- rxn_id: Reaction index

Usage:
    python -m motsart.learning.results_to_data_pkl --help
"""

import random
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

from motsart.common import PathHandler
from .utils import get_guesses_gt_pos_from_files
from goflow.preprocessing import process_reaction_data


# ===================== Default Configuration =====================
DEFAULT_RESULTS_FOLDER = 'results_musica/results_musica'
DEFAULT_RXN_CSV = 'data/cyclo32_atom_mapped_small.csv'
DEFAULT_OUT_DIR = '/Users/leo/Documents/motsart_project/goflow/data/CYCLO/processed_data'
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


# ===================== Helper Functions =====================

def load_guess_gt_samples(path_handler: PathHandler, check_irc_success: bool = True) -> list:
    """
    Load all guess/ground-truth TS pairs with associated metadata.

    Args:
        path_handler: PathHandler configured with results folder and reaction CSV.
        check_irc_success: If True, only include samples with successful IRC.

    Returns:
        List of tuples ``(guess_file, gt_file, rxn_id, rxn_smiles, mol_name)``.
    """
    guess_gt_pairs = path_handler.get_ts_guesses_and_respective_ts_gt(
        check_irc_success=check_irc_success
    )
    samples = []
    for guess_f, gt_f in guess_gt_pairs:
        rxn_id, rxn_smiles = path_handler.get_rxn_id_and_smiles_given_mol_filepath(gt_f)
        samples.append((guess_f, gt_f, rxn_id, rxn_smiles, gt_f.name))
    return samples


def main(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Configuration:")
    print(f"  results_folder:    {args.results_folder}")
    print(f"  rxn_csv:           {args.rxn_csv}")
    print(f"  out_dir:           {args.out_dir}")
    print(f"  check_irc_success: {args.check_irc_success}")
    print(f"  group_by_rxn:      {args.group_by_rxn}")
    print(f"  seed:              {args.seed}")
    print()

    # Initialize PathHandler
    path_handler = PathHandler(
        rxn_csv=Path(args.rxn_csv),
        results_folder=args.results_folder
    )

    # Load feature dictionary
    with open(path_handler.learning_feat_dict, 'rb') as f:
        feat_dict = pickle.load(f)

    # Load and shuffle samples
    samples = load_guess_gt_samples(path_handler, check_irc_success=args.check_irc_success)
    random.shuffle(samples)
    print(f"Found {len(samples)} guess/GT pairs")

    # Extract positions from files
    guess_gt_file_pairs = [(guess_f, gt_f) for guess_f, gt_f, _, _, _ in samples]
    guess_L_N_3, gt_L_N_3 = get_guesses_gt_pos_from_files(guess_gt_file_pairs)

    # Compute split boundaries
    unique_rxn_ids = list({rxn_id for _, _, rxn_id, _, _ in samples})
    random.shuffle(unique_rxn_ids)
    print(f"Num unique reactions: {len(unique_rxn_ids)}")

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    r_train = args.train_ratio / total_ratio
    r_val = args.val_ratio / total_ratio

    if args.group_by_rxn:
        n_train = int(r_train * len(unique_rxn_ids))
        n_val = int(r_val * len(unique_rxn_ids))

        train_rxn_ids = set(unique_rxn_ids[:n_train])
        val_rxn_ids = set(unique_rxn_ids[n_train:n_train + n_val])
        test_rxn_ids = set(unique_rxn_ids[n_train + n_val:])

    # Process samples into data objects
    train_data_L, val_data_L, test_data_L = [], [], []

    for i, (_, _, rxn_id, rxn_smiles, mol_name) in enumerate(tqdm(samples, desc="Processing samples")):
        data_obj = process_reaction_data(
            feat_dict, rxn_smiles, rxn_id, guess_L_N_3[i][None], gt_L_N_3[i][None]
        )[0]
        data_obj.mol_name = mol_name
        data_obj.pos_TS = torch.tensor(gt_L_N_3[i]).float()

        if args.group_by_rxn:
            if rxn_id in train_rxn_ids:
                train_data_L.append(data_obj)
            elif rxn_id in val_rxn_ids:
                val_data_L.append(data_obj)
            else:
                test_data_L.append(data_obj)
        else:
            train_data_L.append(data_obj)  # temporary; split below

    # Handle non-grouped splitting
    if not args.group_by_rxn:
        all_data = train_data_L
        random.shuffle(all_data)

        split_train = int(r_train * len(all_data))
        split_val = int((r_train + r_val) * len(all_data))

        train_data_L = all_data[:split_train]
        val_data_L = all_data[split_train:split_val]
        test_data_L = all_data[split_val:]

    assert len(samples) == len(train_data_L) + len(val_data_L) + len(test_data_L)

    print(f"\nProcessing complete:")
    print(f"  Train size: {len(train_data_L)}")
    print(f"  Val size:   {len(val_data_L)}")
    print(f"  Test size:  {len(test_data_L)}")

    # Save pickle files
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'data_train.pkl', 'wb') as f:
        pickle.dump(train_data_L, f)
    with open(out_dir / 'data_val.pkl', 'wb') as f:
        pickle.dump(val_data_L, f)
    with open(out_dir / 'data_test.pkl', 'wb') as f:
        pickle.dump(test_data_L, f)

    print(f"\nSaved to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess MOTSART guess/GT TS pairs into PyG data objects for learning."
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
        "--check_irc_success", action="store_true", default=True,
        help="Only include samples with successful IRC (default: True)"
    )
    parser.add_argument(
        "--no_check_irc_success", action="store_true",
        help="Include samples regardless of IRC success"
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
    if args.no_check_irc_success:
        args.check_irc_success = False
    if args.no_group_by_rxn:
        args.group_by_rxn = False

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
