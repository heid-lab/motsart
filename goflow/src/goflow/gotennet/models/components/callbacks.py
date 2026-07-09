"""Callbacks for the GotenNet model."""
from random import random
import time
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Optional, Tuple
import pandas as pd

import lightning.pytorch as L
import torch
from lightning import Trainer
from torch_geometric.data import Data

from goflow.flow_matching.utils import compute_steric_clash_penalty, pred_atom_index_align, calc_DMAE, pred_atom_index_align_mad, match_and_compute_rmsd
from goflow.gotennet.utils import RankedLogger
from scipy.spatial.distance import cdist
from rdkit import Chem

from dataclasses import dataclass
import tempfile
import subprocess
import numpy as np
import re


# Conversion constants
HARTREE_TO_KCAL = 627.509
BOHR_TO_ANGSTROM = 0.529177
ANGSTROM_TO_BOHR = 1.88973


@dataclass
class ForceCalculationResult:
    """Result of force calculation."""
    forces: np.ndarray  # Shape (N, 3), in Hartree/Bohr
    energy: float  # In Hartree
    mean_force_norm: float  # In Hartree/Bohr
    max_force_norm: float  # In Hartree/Bohr


# ---------------------------------------------------------------------------
# XTB Backend
# ---------------------------------------------------------------------------

_PERIODIC_TABLE = Chem.GetPeriodicTable()


def _atomic_number_to_symbol(z: int) -> str:
    """Convert atomic number to element symbol."""
    return _PERIODIC_TABLE.GetElementSymbol(int(z))


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

    # First line after $grad contains cycle, energy, gnorm
    # Format can be either:
    #   "cycle 1 -10.123 0.456" (old Turbomole style)
    #   "cycle =      1    SCF energy =    -10.123   |dE/dxyz| =  0.456" (XTB style)
    header_line = lines[grad_start]
    header_parts = header_line.split()
    if len(header_parts) < 2:
        raise ValueError(f"Invalid gradient header: {header_line}")

    # Try to find energy - handle XTB format with "energy =" pattern
    energy = None
    if 'energy' in header_line.lower():
        # XTB format: look for value after "energy ="
        match = re.search(r'energy\s*=\s*([-+]?\d+\.?\d*(?:[eEdD][-+]?\d+)?)', header_line, re.IGNORECASE)
        if match:
            energy = float(match.group(1).replace('D', 'E').replace('d', 'e'))

    if energy is None:
        # Fallback: assume old format where energy is second token
        try:
            energy = float(header_parts[1].replace('D', 'E').replace('d', 'e'))
        except ValueError:
            raise ValueError(f"Could not parse energy from gradient header: {header_line}")

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


def compute_forces(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    charge: int,
    multiplicity: int,
    xtb_path: Optional[str] = None,
    solvent: Optional[str] = None,
) -> ForceCalculationResult:
    """Compute forces using XTB."""
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


def get_charge_from_reaction_smiles(reaction_smiles: str) -> int:
    """
    Calculate the total formal charge from the reactant side of a reaction SMILES.

    Args:
        reaction_smiles: Reaction SMILES in format "R>>P"

    Returns:
        Total formal charge of the reactant molecule(s)
    """
    # Split reaction SMILES to get reactant side
    reactant_smiles = reaction_smiles.split(">>")[0]

    # Parse the reactant SMILES (may contain multiple molecules separated by '.')
    mol = Chem.MolFromSmiles(reactant_smiles)
    if mol is None:
        return 0  # Default to neutral if parsing fails

    # Sum up formal charges of all atoms
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    return total_charge

import itertools
from collections import defaultdict

log = RankedLogger(__name__, rank_zero_only=True)


# --------------------------- Metric Code Start ---------------------------

def build_connectivity(edge_index: torch.Tensor) -> dict:
    """
    Build a connectivity dictionary from an edge_index tensor.
    
    Parameters:
        edge_index (torch.Tensor): Tensor of shape (2, E) representing bond connections.
    
    Returns:
        dict: A dictionary mapping each atom index to a sorted list of its neighbors.
    """
    connectivity = defaultdict(set)
    
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    
    for i, j in zip(src, dst):
        connectivity[i].add(j)
        connectivity[j].add(i)
    
    connectivity = {node: sorted(neigh) for node, neigh in connectivity.items()}
    return connectivity


def extract_angles_from_connectivity(connectivity: dict) -> torch.Tensor:
    """
    Extract angle (triplet) indices given the connectivity dictionary.
    
    For each central atom j that has at least two neighbors,
    form (i, j, k) for every unique pair {i, k}.

    Parameters:
        connectivity (dict): Dictionary mapping node -> list of neighbors.
    
    Returns:
        torch.Tensor: Tensor of shape (num_angles, 3) where each row is (i, j, k) with j as the vertex.
    """
    angles = []
    for j, neighbors in connectivity.items():
        if len(neighbors) < 2:
            continue
        # Use combinations to generate all unique pairs
        for i, k in itertools.combinations(neighbors, 2):
            angles.append([i, j, k])
    if angles:
        return torch.tensor(angles, dtype=torch.long)
    else:
        return torch.empty((0, 3), dtype=torch.long)


def extract_dihedrals_from_connectivity(connectivity: dict) -> torch.Tensor:
    """
    Extract dihedral (quadruplet) indices from the connectivity dictionary.
    
    For every bond (j,k), for every neighbor i of j (excluding k)
    and every neighbor l of k (excluding j), form (i, j, k, l).

    Parameters:
        connectivity (dict): Dictionary mapping node -> list of neighbors.
    
    Returns:
        torch.Tensor: Tensor of shape (num_dihedrals, 4) with each row being (i, j, k, l).
    """
    dihedrals = []
    for j, neighbors_j in connectivity.items():
        for k in neighbors_j:
            # For bond (j, k), iterate over neighbors of j and k, excluding the counterpart.
            for i in neighbors_j:
                if i == k:
                    continue
                # Get neighbors of k; if none, skip.
                neighbors_k = connectivity.get(k, [])
                for l in neighbors_k:
                    if l == j:
                        continue
                    dihedrals.append([i, j, k, l])
    if dihedrals:
        return torch.tensor(dihedrals, dtype=torch.long)
    else:
        return torch.empty((0, 4), dtype=torch.long)


def compute_bond_angles(
    coords_N_3: torch.Tensor, angle_indices_M_3: torch.Tensor
) -> torch.Tensor:
    """
    Compute the bond angles for a set of triplets. Each triplet is assumed
    to be (i, j, k) with j as the vertex. The bond angle is computed as

        θ = arccos [ (vec1 · vec2) / (||vec1|| ||vec2||) ]

    Parameters:
        coords_N_3 (torch.Tensor): Tensor of shape (N,3) with 3D positions.
        angle_indices_M_3 (torch.Tensor): Tensor of shape (M,3) with atom indices
                                      defining each angle = (i, j, k).

    Returns:
        torch.Tensor: A tensor of shape (M,) with the angles in radians.
    """
    vec1_M_3 = coords_N_3[angle_indices_M_3[:, 0]] - coords_N_3[angle_indices_M_3[:, 1]]
    vec2_M_3 = coords_N_3[angle_indices_M_3[:, 2]] - coords_N_3[angle_indices_M_3[:, 1]]
    dot_prod_M = (vec1_M_3 * vec2_M_3).sum(dim=1)
    norm1_M = vec1_M_3.norm(dim=1)
    norm2_M = vec2_M_3.norm(dim=1)
    cosine_M = dot_prod_M / (norm1_M * norm2_M + 1e-9)
    # Clamp to [-1,1] to avoid numerical issues with arccos
    cosine_M = torch.clamp(cosine_M, -1.0, 1.0)
    angles_M = torch.acos(cosine_M)
    return angles_M


def compute_dihedral_angles(
    coords_N_3: torch.Tensor, dihedral_indices_M_4: torch.Tensor
) -> torch.Tensor:
    """
    Compute the dihedral (torsion) angles for a set of quadruplets.
    A dihedral is defined by four atoms with indices (i, j, k, l).

    Parameters:
        coords_N_3 (torch.Tensor): Tensor of shape (N,3) with 3D positions.
        dihedral_indices_M_4 (torch.Tensor): Tensor of shape (M,4) with atom indices
                                           defining each dihedral.

    Returns:
        torch.Tensor: A tensor of shape (M,) with the dihedral angles in radians.
    """
    p0_M_3 = coords_N_3[dihedral_indices_M_4[:, 0]]
    p1_M_3 = coords_N_3[dihedral_indices_M_4[:, 1]]
    p2_M_3 = coords_N_3[dihedral_indices_M_4[:, 2]]
    p3_M_3 = coords_N_3[dihedral_indices_M_4[:, 3]]

    b0_M_3 = p1_M_3 - p0_M_3
    b1_M_3 = p2_M_3 - p1_M_3
    b2_M_3 = p3_M_3 - p2_M_3

    n1_M_3 = torch.cross(b0_M_3, b1_M_3, dim=1)
    n2_M_3 = torch.cross(b1_M_3, b2_M_3, dim=1)

    n1_norm_M_3 = n1_M_3 / (n1_M_3.norm(dim=1, keepdim=True) + 1e-9)
    n2_norm_M_3 = n2_M_3 / (n2_M_3.norm(dim=1, keepdim=True) + 1e-9)
    b1_unit_M_3 = b1_M_3 / (b1_M_3.norm(dim=1, keepdim=True) + 1e-9)

    m1_M_3 = torch.cross(n1_norm_M_3, b1_unit_M_3, dim=1)

    x_M = (n1_norm_M_3 * n2_norm_M_3).sum(dim=1)
    y_M = (m1_M_3 * n2_norm_M_3).sum(dim=1)

    dihedral_angles_M = torch.atan2(y_M, x_M)
    return dihedral_angles_M


def evaluate_geometry(
    data: Data,
    r_threshold: float = .8,
    epsilon: float = 1.0,
) -> Dict[str, float]:
    """
    Parameters:
    data (torch_geometric.data.Data): Reaction data
    r_threshold (float): Distance threshold for steric clash penalty.
    epsilon (float): Scaling factor for the clash penalty.

    Returns:
    Dict[str, float]
    """    
    # RMSE error    
    rmse = match_and_compute_rmsd(data)

    # MAE error
    pred_pos_N_3, gt_pos_N_3 = pred_atom_index_align(data.smiles, data.pos, data.pos_gen)
    pred_pos_aligned_mae = pred_atom_index_align_mad(data.smiles, data.pos, data.pos_gen)
    mae = calc_DMAE(cdist(gt_pos_N_3, gt_pos_N_3), cdist(pred_pos_aligned_mae, pred_pos_aligned_mae))

    connectivity = build_connectivity(data.edge_index)
    # Extract angle and dihedral indices from the connectivity.
    angle_indices_M1_3 = extract_angles_from_connectivity(connectivity)
    dihedral_indices_M2_4 = extract_dihedrals_from_connectivity(connectivity)

    # Bond angle comparison.
    gt_angles_M1 = compute_bond_angles(gt_pos_N_3, angle_indices_M1_3)
    pred_angles_M1 = compute_bond_angles(pred_pos_N_3, angle_indices_M1_3)
    # Convert radians to degrees.
    bond_angle_error = (torch.abs(gt_angles_M1 - pred_angles_M1) * 180.0 / math.pi).mean()

    # Dihedral angle comparison.
    gt_dihedrals_M2 = compute_dihedral_angles(gt_pos_N_3, dihedral_indices_M2_4)
    pred_dihedrals_M2 = compute_dihedral_angles(pred_pos_N_3, dihedral_indices_M2_4)
    diff_M2 = torch.abs(gt_dihedrals_M2 - pred_dihedrals_M2)
    # Handle periodicity: if the difference is larger than pi, wrap around.
    diff_M2 = torch.where(diff_M2 > math.pi, 2 * math.pi - diff_M2, diff_M2)
    dihedral_angle_error = (diff_M2 * 180.0 / math.pi).mean()

    # Steric clash penalty
    steric_clash_pred = compute_steric_clash_penalty(pred_pos_N_3, r_threshold, epsilon)
    steric_clash_gt = compute_steric_clash_penalty(gt_pos_N_3, r_threshold, epsilon)
    steric_clash_diff = (steric_clash_pred - steric_clash_gt).item()
    steric_clash_diff = min(steric_clash_diff, 9999)

    result = {
        "mae": round(float(mae), 4),
        "rmse": round(rmse.item(), 4),
        "angle_error": round(bond_angle_error.item(), 4),
        "dihedral_error": round(dihedral_angle_error.item(), 4),
        "steric_clash": round(steric_clash_diff, 4)
    }

    # Force norm evaluation (optional, enabled via config)
    if getattr(data, 'evaluate_force_norm', False):
        try:
            # Get atomic numbers from data.atom_type
            atomic_numbers = data.atom_type.cpu().numpy()
            # Get predicted coordinates (already aligned)
            coords = pred_pos_N_3.cpu().numpy() if isinstance(pred_pos_N_3, torch.Tensor) else pred_pos_N_3

            charge = get_charge_from_reaction_smiles(data.smiles)
            force_result = compute_forces(
                coords=coords,
                atomic_numbers=atomic_numbers,
                charge=charge,
                multiplicity=getattr(data, 'force_multiplicity', 1),
            )

            result["mean_force_norm"] = round(float(force_result.mean_force_norm), 6)
            result["max_force_norm"] = round(float(force_result.max_force_norm), 6)
        except Exception as e:
            log.warning(f"Force computation failed: {e}")
            result["mean_force_norm"] = None
            result["max_force_norm"] = None

    return result

# --------------------------- Metric Code End ---------------------------


class TestAndSaveResultsAfterTrainingCallback(L.Callback):
    def __init__(self, save_path, runs_stats_path=None):#, mr_stats_path=None):
        self.save_path = Path(save_path)
        self.runs_stats_path = Path(runs_stats_path)
        self._test_start_time = None

    def save_test_predictions(self, module):
        pickle_save_path = self.save_path / 'test_samples/samples_all.pkl'
        pickle_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_save_path, "wb") as f:
            pickle.dump(module.test_results_C, f)

    def save_stats_to_csv(self, pd_results_mean, module):
        mr_stats_file = self.runs_stats_path / 'stats.csv'
        
        pd_results_mean['num_steps'] = module.num_steps
        pd_results_mean['num_samples'] = module.num_samples
        
        if mr_stats_file.exists():
            df = pd.read_csv(mr_stats_file)
            df = pd.concat([df, pd.DataFrame([pd_results_mean])], ignore_index=True)
            df.to_csv(mr_stats_file, index=False, float_format='%.3f')
        else:
            pd.DataFrame([pd_results_mean]).to_csv(mr_stats_file, index=False, float_format='%.3f')

    def on_test_start(self, trainer: Trainer, module: L.LightningModule):
        module.test_results_C = []
        self._test_start_time = time.perf_counter()
    
    def on_test_end(self, trainer: Trainer, module):
        inference_time_per_rxn = (time.perf_counter() - self._test_start_time) / len(module.test_results_C)
        for data in module.test_results_C:
            data.avg_inference_time = inference_time_per_rxn

        self.save_test_predictions(module)


class EMALossCallback(L.Callback):
    """
    Exponential Moving Average (EMA) Loss Callback.
    This callback calculates and logs the EMA of the validation loss.
    """

    def __init__(
            self,
            alpha: float = 0.99,
            soft_beta: float = 10,
            validation_loss_name: str = "val_loss",
            ema_log_name: str = "validation/ema_loss"
    ):
        """
        Initialize the EMALossCallback.

        Args:
            alpha (float): The decay factor for EMA calculation. Default is 0.99.
            soft_beta (float): The soft beta factor for loss capping. Default is 10.
            validation_loss_name (str): The name of the validation loss in the outputs. Default is "val_loss".
            ema_log_name (str): The name under which to log the EMA loss. Default is "validation/ema_loss".
        """
        super().__init__()
        self.alpha = alpha
        self.ema: Optional[torch.Tensor] = None
        self.num_batches: int = 0
        self.soft_beta = soft_beta
        self.total_loss: Optional[torch.Tensor] = None
        self.validation_loss_name = validation_loss_name
        self.ema_log_name = ema_log_name

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dictionary.

        Args:
            state_dict (Dict[str, Any]): The state dictionary to load from.
        """
        if "ema_loss" in state_dict:
            log.info("EMA loss loaded")
            self.ema = state_dict["ema_loss"]
        else:
            log.info("EMA loss not found in checkpoint")
            self.ema = None

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state dictionary.

        Returns:
            Dict[str, Any]: The state dictionary containing the EMA loss.
        """
        return {"ema_loss": self.ema}

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Called when the validation epoch begins.

        Args:
            trainer (L.Trainer): The trainer instance.
            pl_module (L.LightningModule): The LightningModule instance.
        """
        self.num_batches = 0
        self.total_loss = torch.tensor(0.0, device=pl_module.device)
        if self.ema is not None and isinstance(self.ema, torch.Tensor):
            self.ema = self.ema.to(pl_module.device)

    def on_validation_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            **kwargs
    ) -> None:
        """
        Called when a validation batch ends.

        Args:
            trainer (L.Trainer): The trainer instance.
            pl_module (L.LightningModule): The LightningModule instance.
            outputs (Any): The outputs from the validation step.
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.
            **kwargs: Additional keyword arguments.
        """
        self.total_loss += outputs
        self.num_batches += 1

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Called when the validation epoch ends.

        Args:
            trainer (L.Trainer): The trainer instance.
            pl_module (L.LightningModule): The LightningModule instance.
        """
        avg_loss = self.total_loss / self.num_batches
        if self.ema is None:
            self.ema = avg_loss
        else:
            if self.soft_beta is not None:
                avg_loss = torch.min(torch.stack([avg_loss, self.ema * self.soft_beta]))
            self.ema = self.alpha * self.ema + (1 - self.alpha) * avg_loss
        pl_module.log(self.ema_log_name, self.ema, on_step=False, on_epoch=True)
