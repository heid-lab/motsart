"""Biased AFIR optimization backends.

The xTB backend preserves the original implementation. The MLIP backend uses
ASE optimization on ``E_total = E_MLIP + E_AFIR_bias`` and returns the same path
arrays consumed by :mod:`afir_path_guesser`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read as ase_read
from ase.optimize import FIRE

from motsart.complex_finder.utils import (
    ReactionData,
    read_energy_coords_file,
    xtb_optimize_with_applied_potentials,
)
from motsart.validator.orca_validator.orca_external_tools.mlip_external import (
    _build_atoms,
    get_calculator,
    resolve_device,
)


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.52917721067
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG


def afir_bias_energy_forces(
    coords_N_3: np.ndarray,
    constraints: dict[tuple[int, int], float],
    force_constant: float,
) -> tuple[float, np.ndarray]:
    """Return AFIR harmonic bias energy in eV and forces in eV/Angstrom."""
    coords_N_3 = np.asarray(coords_N_3, dtype=float)
    forces_N_3 = np.zeros_like(coords_N_3)
    energy_eh = 0.0

    for (idx_1, idx_2), target_ang in constraints.items():
        delta_vec = coords_N_3[idx_1] - coords_N_3[idx_2]
        dist_ang = float(np.linalg.norm(delta_vec))
        if dist_ang < 1e-12:
            continue

        dist_bohr = dist_ang * ANG_TO_BOHR
        target_bohr = target_ang * ANG_TO_BOHR
        delta_bohr = dist_bohr - target_bohr
        energy_eh += force_constant * delta_bohr**2

        grad_eh_ang = 2.0 * force_constant * delta_bohr * ANG_TO_BOHR * (delta_vec / dist_ang)
        force_ev_ang = -grad_eh_ang * HARTREE_TO_EV
        forces_N_3[idx_1] += force_ev_ang
        forces_N_3[idx_2] -= force_ev_ang

    return energy_eh * HARTREE_TO_EV, forces_N_3


class AFIRBiasedMLIPCalculator(Calculator):
    """ASE calculator adding a harmonic AFIR distance bias to an MLIP PES."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, constraints: dict[tuple[int, int], float], force_constant: float):
        super().__init__()
        self.base_calc = base_calc
        self.constraints = constraints
        self.force_constant = force_constant

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        base_atoms = atoms.copy()
        base_atoms.calc = self.base_calc
        base_energy_ev = float(base_atoms.get_potential_energy())
        base_forces_ev_ang = np.asarray(base_atoms.get_forces(), dtype=float)

        bias_energy_ev, bias_forces_ev_ang = afir_bias_energy_forces(
            atoms.get_positions(),
            self.constraints,
            self.force_constant,
        )

        self.results["energy"] = base_energy_ev + bias_energy_ev
        self.results["forces"] = base_forces_ev_ang + bias_forces_ev_ang


def run_xtb_afir_optimization(
    rxn_data: ReactionData,
    constraints: dict[tuple[int, int], float],
    reactive_complex_xyz_file: Path,
    force_constant: float,
    output_dir: Path,
) -> tuple[np.ndarray, list[np.ndarray], list[list[str]]]:
    """Run the original xTB constrained optimization and parse its path log."""
    log_file = xtb_optimize_with_applied_potentials(
        rxn_data,
        constraints,
        reactive_complex_xyz_file,
        fc=force_constant,
        output_dir=output_dir,
    )
    return read_energy_coords_file(log_file)


def run_mlip_afir_optimization(
    rxn_data: ReactionData,
    constraints: dict[tuple[int, int], float],
    reactive_complex_xyz_file: Path,
    force_constant: float,
    model: str,
    task: Optional[str],
    device: Optional[str],
    fmax: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Run AFIR biased optimization with an OMol25/FAIRChem MLIP through ASE."""

    initial_atoms = ase_read(str(reactive_complex_xyz_file))
    atoms = _build_atoms(
        initial_atoms.get_chemical_symbols(),
        initial_atoms.get_positions(),
        charge=rxn_data.charge,
        mult=1,
    )

    device = resolve_device(device)
    base_calc = get_calculator(model, device, task)
    atoms.calc = AFIRBiasedMLIPCalculator(base_calc, constraints, force_constant)

    energies_eh: list[float] = []
    coords: list[np.ndarray] = []
    symbols: list[list[str]] = []

    def record_step() -> None:
        energies_eh.append(float(atoms.get_potential_energy()) / HARTREE_TO_EV)
        coords.append(np.asarray(atoms.get_positions(), dtype=float).copy())
        symbols.append(list(atoms.get_chemical_symbols()))

    record_step()
    opt = FIRE(atoms, logfile=None)
    opt.attach(record_step, interval=1)
    opt.run(fmax=fmax, steps=steps)

    return np.asarray(energies_eh), np.asarray(coords), symbols
