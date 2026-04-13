"""Force computation backends and adjacency-matrix utilities for validation.

Provides a unified :func:`compute_forces` interface that dispatches to
xTB, ORCA DFT, or PySCF (CPU/GPU) backends, plus an RDKit adjacency
matrix helper used for IRC bond-change comparison.
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import tempfile
import subprocess
import numpy as np
from rdkit import Chem


# Conversion constants
HARTREE_TO_KCAL = 627.509
BOHR_TO_ANGSTROM = 0.529177
ANGSTROM_TO_BOHR = 1.88973


class ForceBackend(Enum):
    """Available backends for force calculation."""
    XTB = "xtb"
    ORCA_DFT = "orca_dft"
    PYSCF = "pyscf"
    PYSCF_GPU = "pyscf_gpu"


@dataclass
class ForceCalculationResult:
    """Result of force calculation."""
    forces: np.ndarray  # Shape (N, 3), in Hartree/Bohr
    energy: float  # In Hartree
    mean_force_norm: float  # In Hartree/Bohr
    max_force_norm: float  # In Hartree/Bohr


def compute_forces(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    backend: ForceBackend = ForceBackend.XTB,
    charge: int = 0,
    multiplicity: int = 1,
    # PySCF-specific options
    basis: str = "def2-SVP",
    functional: str = "b3lyp",
    # XTB/ORCA-specific options
    xtb_path: Optional[str] = None,
    orca_path: Optional[str] = None,
    solvent: Optional[str] = None,
    nprocs: int = 1,
    maxcore: int = 2048,
) -> ForceCalculationResult:
    """
    Compute forces on atoms using specified backend.

    Args:
        coords: Atomic coordinates, shape (N, 3), in Angstrom
        atomic_numbers: Atomic numbers, shape (N,)
        backend: Calculation backend (XTB, ORCA_DFT, PYSCF, PYSCF_GPU)
        charge: Total charge of the system
        multiplicity: Spin multiplicity (1 = singlet, 2 = doublet, etc.)
        basis: Basis set for DFT calculations (PySCF/ORCA)
        functional: DFT functional (PySCF/ORCA)
        xtb_path: Path to XTB executable (for XTB backend)
        orca_path: Path to ORCA executable (for ORCA backend)
        solvent: Solvent name for implicit solvation (optional)
        nprocs: Number of processors (ORCA/PySCF)
        maxcore: Memory per core in MB (ORCA)

    Returns:
        ForceCalculationResult with forces, energy, and force statistics
    """
    coords = np.asarray(coords, dtype=np.float64)
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int32)

    if coords.shape[0] != atomic_numbers.shape[0]:
        raise ValueError(f"Mismatch: coords has {coords.shape[0]} atoms, "
                        f"atomic_numbers has {atomic_numbers.shape[0]}")

    if coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {coords.shape}")

    if backend == ForceBackend.XTB:
        return _compute_forces_xtb(
            coords, atomic_numbers, charge, multiplicity,
            xtb_path=xtb_path, solvent=solvent
        )
    elif backend == ForceBackend.ORCA_DFT:
        return _compute_forces_orca_dft(
            coords, atomic_numbers, charge, multiplicity,
            orca_path=orca_path, basis=basis, functional=functional,
            solvent=solvent, nprocs=nprocs, maxcore=maxcore
        )
    elif backend in (ForceBackend.PYSCF, ForceBackend.PYSCF_GPU):
        use_gpu = (backend == ForceBackend.PYSCF_GPU)
        return _compute_forces_pyscf(
            coords, atomic_numbers, charge, multiplicity,
            basis=basis, functional=functional, use_gpu=use_gpu,
            solvent=solvent, nprocs=nprocs
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# XTB Backend
# ---------------------------------------------------------------------------

# Complete periodic table (Z=1 to Z=118)
ATOMIC_SYMBOLS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
    30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
    37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
    65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
    72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
    79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
    86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
    93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es',
    100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db',
    106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg',
    112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og',
}


def _atomic_number_to_symbol(z: int) -> str:
    """Convert atomic number to element symbol."""
    if z not in ATOMIC_SYMBOLS:
        raise ValueError(f"Unknown atomic number: {z}. Valid range is 1-118.")
    return ATOMIC_SYMBOLS[z]


def _write_xyz_file(path: Path, coords: np.ndarray, atomic_numbers: np.ndarray,
                    comment: str = "") -> None:
    """Write XYZ file from coordinates and atomic numbers."""
    n_atoms = len(atomic_numbers)
    with open(path, 'w') as f:
        f.write(f"{n_atoms}\n")
        f.write(f"{comment}\n")
        for i in range(n_atoms):
            symbol = _atomic_number_to_symbol(atomic_numbers[i])
            f.write(f"{symbol:2s} {coords[i, 0]:15.8f} {coords[i, 1]:15.8f} {coords[i, 2]:15.8f}\n")


def _parse_xtb_gradient(gradient_file: Path, n_atoms: int) -> Tuple[float, np.ndarray]:
    """
    Parse XTB gradient file (Turbomole format) to extract energy and gradients.

    The Turbomole gradient format is:
        $grad
        cycle X  energy  gnorm
        x1 y1 z1 element1
        x2 y2 z2 element2
        ...
        gx1 gy1 gz1
        gx2 gy2 gz2
        ...
        $end

    Args:
        gradient_file: Path to gradient file
        n_atoms: Expected number of atoms (for validation)

    Returns:
        Tuple of (energy in Hartree, gradients shape (N, 3) in Hartree/Bohr)
    """
    with open(gradient_file, 'r') as f:
        lines = f.readlines()

    # Find the LAST $grad section (in case multiple exist)
    grad_start = None
    for i, line in enumerate(lines):
        if '$grad' in line.lower():
            grad_start = i + 1

    if grad_start is None:
        raise ValueError("Could not find $grad section in gradient file")

    # First line after $grad contains: cycle, energy, gnorm
    header_parts = lines[grad_start].split()
    if len(header_parts) < 2:
        raise ValueError(f"Invalid gradient header: {lines[grad_start]}")
    energy = float(header_parts[1])

    # Turbomole format: N coordinate lines, then N gradient lines
    # Coordinate lines: x y z element (4 columns)
    # Gradient lines: gx gy gz (3 columns)
    coord_start = grad_start + 1
    grad_line_start = coord_start + n_atoms

    # Parse gradient lines (exactly n_atoms lines after coordinates)
    gradients = []
    for i in range(n_atoms):
        line_idx = grad_line_start + i
        if line_idx >= len(lines):
            raise ValueError(f"Gradient file truncated: expected {n_atoms} gradient lines")

        line = lines[line_idx].strip()
        if '$end' in line.lower():
            raise ValueError(f"Unexpected $end before all gradients parsed (got {i}/{n_atoms})")

        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid gradient line {line_idx}: {line}")

        # Handle Fortran D-notation if present (e.g., 1.0D-05 -> 1.0E-05)
        gx = float(parts[0].replace('D', 'E').replace('d', 'e'))
        gy = float(parts[1].replace('D', 'E').replace('d', 'e'))
        gz = float(parts[2].replace('D', 'E').replace('d', 'e'))
        gradients.append([gx, gy, gz])

    gradients = np.array(gradients)

    # Validate shape
    if gradients.shape != (n_atoms, 3):
        raise ValueError(
            f"Gradient shape mismatch: expected ({n_atoms}, 3), got {gradients.shape}"
        )

    return energy, gradients


def _compute_forces_xtb(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    charge: int,
    multiplicity: int,
    xtb_path: Optional[str] = None,
    solvent: Optional[str] = None,
) -> ForceCalculationResult:
    """Compute forces using XTB (direct call)."""
    xtb_cmd = xtb_path or "xtb"
    n_atoms = len(atomic_numbers)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        xyz_file = tmpdir / "input.xyz"

        # Write input file
        _write_xyz_file(xyz_file, coords, atomic_numbers)

        # Build XTB command
        cmd = [
            xtb_cmd,
            str(xyz_file),
            "--gfn", "2",
            "--grad",
            "--chrg", str(charge),
            "--uhf", str(multiplicity - 1),
        ]

        if solvent:
            cmd.extend(["--alpb", solvent])

        # Run XTB
        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"XTB failed:\n{result.stderr}")

        # Parse gradient file
        gradient_file = tmpdir / "gradient"
        if not gradient_file.exists():
            raise RuntimeError("XTB did not produce gradient file")

        energy, gradients = _parse_xtb_gradient(gradient_file, n_atoms)

    # Forces = -gradients
    forces = -gradients

    # Compute statistics
    force_norms = np.linalg.norm(forces, axis=1)
    mean_force_norm = np.mean(force_norms)
    max_force_norm = np.max(force_norms)

    return ForceCalculationResult(
        forces=forces,
        energy=energy,
        mean_force_norm=mean_force_norm,
        max_force_norm=max_force_norm
    )


# ---------------------------------------------------------------------------
# ORCA DFT Backend
# ---------------------------------------------------------------------------

ORCA_ENGRAD_TEMPLATE = """! {FUNCTIONAL} {BASIS} D4 def2/J RIJCOSX EnGrad
! NoUseSym

%maxcore {MAXCORE}

%pal nprocs {NPROCS} end

{SOLVENT_BLOCK}

* xyz {CHARGE} {MULT}
{XYZ_BLOCK}
*
"""

ORCA_SOLVENT_BLOCK = """%cpcm
    smd true
    smdsolvent "{SOLVENT}"
end
"""


def _parse_fortran_float(s: str) -> float:
    """Parse a float that may use Fortran D-notation (e.g., 1.0D-05)."""
    return float(s.replace('D', 'E').replace('d', 'e'))


def _parse_orca_engrad(engrad_file: Path, expected_n_atoms: int) -> Tuple[float, np.ndarray]:
    """
    Parse ORCA .engrad file to extract energy and gradients.

    Args:
        engrad_file: Path to .engrad file
        expected_n_atoms: Expected number of atoms (for validation)

    Returns:
        Tuple of (energy in Hartree, gradients shape (N, 3) in Hartree/Bohr)
    """
    with open(engrad_file, 'r') as f:
        lines = f.readlines()

    n_atoms = None
    energy = None
    gradients = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Number of atoms
        if line.startswith("# Number of atoms"):
            i += 1
            if i >= len(lines):
                raise ValueError("engrad file truncated after '# Number of atoms'")
            n_atoms = int(lines[i].strip())

        # Energy
        elif line.startswith("# The current total energy"):
            i += 1
            if i >= len(lines):
                raise ValueError("engrad file truncated after '# The current total energy'")
            energy = _parse_fortran_float(lines[i].strip())

        # Gradients
        elif line.startswith("# The current gradient"):
            if n_atoms is None:
                raise ValueError(
                    "engrad file has gradient section before atom count; cannot parse"
                )
            i += 1
            for _ in range(n_atoms * 3):
                if i >= len(lines):
                    raise ValueError("engrad file truncated in gradient section")
                gradients.append(_parse_fortran_float(lines[i].strip()))
                i += 1
            continue

        i += 1

    if energy is None:
        raise ValueError("Could not find energy in ORCA engrad file")
    if len(gradients) == 0:
        raise ValueError("Could not find gradients in ORCA engrad file")
    if n_atoms is None:
        raise ValueError("Could not find atom count in ORCA engrad file")

    gradients = np.array(gradients).reshape(-1, 3)

    # Validate shape
    if gradients.shape != (expected_n_atoms, 3):
        raise ValueError(
            f"Gradient shape mismatch: expected ({expected_n_atoms}, 3), "
            f"got {gradients.shape} (file reported {n_atoms} atoms)"
        )

    return energy, gradients


def _compute_forces_orca_dft(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    charge: int,
    multiplicity: int,
    orca_path: Optional[str] = None,
    basis: str = "def2-SVP",
    functional: str = "B3LYP",
    solvent: Optional[str] = None,
    nprocs: int = 1,
    maxcore: int = 2048,
) -> ForceCalculationResult:
    """Compute forces using ORCA DFT."""
    orca_cmd = orca_path or "orca"
    n_atoms = len(atomic_numbers)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        inp_file = tmpdir / "input.inp"
        out_file = tmpdir / "input.out"
        engrad_file = tmpdir / "input.engrad"

        # Build XYZ block
        xyz_lines = []
        for i in range(n_atoms):
            symbol = _atomic_number_to_symbol(atomic_numbers[i])
            xyz_lines.append(
                f"  {symbol:2s} {coords[i, 0]:15.8f} {coords[i, 1]:15.8f} {coords[i, 2]:15.8f}"
            )
        xyz_block = "\n".join(xyz_lines)

        # Build solvent block
        solvent_block = ""
        if solvent:
            solvent_block = ORCA_SOLVENT_BLOCK.format(SOLVENT=solvent)

        # Write input file
        inp_content = ORCA_ENGRAD_TEMPLATE.format(
            FUNCTIONAL=functional,
            BASIS=basis,
            MAXCORE=maxcore,
            NPROCS=nprocs,
            SOLVENT_BLOCK=solvent_block,
            CHARGE=charge,
            MULT=multiplicity,
            XYZ_BLOCK=xyz_block
        )
        inp_file.write_text(inp_content)

        # Run ORCA
        with open(out_file, 'w') as out_f:
            result = subprocess.run(
                [orca_cmd, str(inp_file.name)],
                cwd=tmpdir,
                stdout=out_f,
                stderr=subprocess.STDOUT,
            )

        if result.returncode != 0 or not engrad_file.exists():
            # Try to get error from output
            if out_file.exists():
                error_msg = out_file.read_text()[-2000:]
            else:
                error_msg = "Unknown error"
            raise RuntimeError(f"ORCA failed:\n{error_msg}")

        # Parse engrad file
        energy, gradients = _parse_orca_engrad(engrad_file, n_atoms)

    # Forces = -gradients
    forces = -gradients

    # Compute statistics
    force_norms = np.linalg.norm(forces, axis=1)
    mean_force_norm = np.mean(force_norms)
    max_force_norm = np.max(force_norms)

    return ForceCalculationResult(
        forces=forces,
        energy=energy,
        mean_force_norm=mean_force_norm,
        max_force_norm=max_force_norm
    )


# ---------------------------------------------------------------------------
# PySCF Backend (CPU and GPU)
# ---------------------------------------------------------------------------

def _compute_forces_pyscf(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    charge: int,
    multiplicity: int,
    basis: str = "def2-SVP",
    functional: str = "b3lyp",
    use_gpu: bool = False,
    solvent: Optional[str] = None,
    nprocs: int = 1,
) -> ForceCalculationResult:
    """Compute forces using PySCF (CPU or GPU)."""
    from pyscf import gto, dft, lib

    n_atoms = len(atomic_numbers)

    # Set number of threads
    lib.num_threads(nprocs)

    # Build molecule
    atom_list = []
    for i in range(n_atoms):
        symbol = _atomic_number_to_symbol(atomic_numbers[i])
        atom_list.append([symbol, coords[i].tolist()])

    mol = gto.Mole()
    mol.atom = atom_list
    mol.basis = basis
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.unit = 'Angstrom'
    mol.build()

    # Set up DFT calculation
    mf = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
    mf.xc = functional

    # Convert to GPU if requested (do this BEFORE solvent wrapper)
    if use_gpu:
        if not hasattr(mf, 'to_gpu'):
            raise RuntimeError(
                "GPU backend unavailable. Install GPU4PySCF: pip install gpu4pyscf"
            )
        try:
            mf = mf.to_gpu()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPU backend: {e}")

    # Add solvent if specified (after GPU conversion)
    if solvent:
        mf = mf.SMD()
        mf.with_solvent.solvent = solvent

    # Run SCF
    energy = mf.kernel()

    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge")

    # Compute gradients
    grad_calculator = mf.nuc_grad_method()
    gradients = grad_calculator.kernel()  # Shape (N, 3), in Hartree/Bohr

    # Validate shape
    if gradients.shape != (n_atoms, 3):
        raise ValueError(
            f"PySCF gradient shape mismatch: expected ({n_atoms}, 3), got {gradients.shape}"
        )

    # Forces = -gradients
    forces = -gradients

    # Compute statistics
    force_norms = np.linalg.norm(forces, axis=1)
    mean_force_norm = np.mean(force_norms)
    max_force_norm = np.max(force_norms)

    return ForceCalculationResult(
        forces=forces,
        energy=energy,
        mean_force_norm=mean_force_norm,
        max_force_norm=max_force_norm
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_adj_matrix_from_mol(mol: Chem.Mol, idx_to_mn=None):
    """
    Returns adjacency matrix from a mol, sorted by Atom Map Number.
    """
    mol.UpdatePropertyCache(strict=False)
    if idx_to_mn is not None:
        mol = Chem.RenumberAtoms(mol, np.argsort(idx_to_mn).tolist())
    return Chem.GetAdjacencyMatrix(mol)
