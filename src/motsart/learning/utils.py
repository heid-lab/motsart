"""I/O helpers for loading TS guess and ground-truth geometries from XYZ files."""

from typing import List, Tuple
import numpy as np
from ase.io import read


def get_guesses_gt_pos_from_files(ts_guess_gt_pairs_B_2: List) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    ts_guess_B_N_3 = [read(guess_file).get_positions() for guess_file, _ in ts_guess_gt_pairs_B_2]
    ts_gt_B_N_3 = [read(gt_file).get_positions() for _, gt_file in ts_guess_gt_pairs_B_2]
    return ts_guess_B_N_3, ts_gt_B_N_3


def get_guess_gt_pos_from_file(ts_guess_gt_pairs_2: List) -> Tuple[np.ndarray, np.ndarray]:
    guess_N_3 = read(ts_guess_gt_pairs_2[0]).get_positions()
    gt_N_3 = read(ts_guess_gt_pairs_2[1]).get_positions()
    return guess_N_3, gt_N_3