"""Shared utilities for path guesser modules."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save_energies_plot(path_energies_I: np.ndarray, ts_guess_i: int, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(path_energies_I * 627.509, 'o-')
    plt.axvline(ts_guess_i, color='r', linestyle='--', label='TS')
    plt.xlabel('Step')
    plt.ylabel('Energy (kcal/mol)')
    plt.title(out_path.stem)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()     