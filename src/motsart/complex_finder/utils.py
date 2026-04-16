"""Utility functions for the complex finder module.

Includes molecular structure I/O, xTB calculations, evolutionary algorithm operators
(rotation/translation mutations), conformer generation, bond analysis, and
constrained geometry optimization.
"""

import time
import os
import random
import subprocess
import tempfile
import shutil
from typing import Any, List, Optional, Set, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor, rdDetermineBonds
from rdkit.Geometry import Point3D
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from ase.io import read
import numpy as np
import io
from dataclasses import dataclass
from pathlib import Path
import qcelemental as qcel
from motsart.conf_default import OptimizationConfig

from motsart.common import PathHandler
from itertools import combinations


HARTREE_TO_KCAL = 627.509
KCAL_TO_HARTREE = 1 / HARTREE_TO_KCAL

ps = Chem.SmilesParserParams()
ps.removeHs = False


def kabsch_align_pairwise_I(pred_I_N_3: np.ndarray, tgt_I_N_3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Align each target frame (tgt[i]) onto the corresponding pred frame (pred[i])
    via Kabsch, vectorized over I.

    Args:
        pred_I_N_3: (I, N, 3)
        tgt_I_N_3:  (I, N, 3)
    Returns:
        aligned_tgt_I_N_3: (I, N, 3)
        rmsd_I: (I,) per-frame RMSD after alignment
    """
    print(pred_I_N_3.shape, tgt_I_N_3.shape)
    assert pred_I_N_3.shape == tgt_I_N_3.shape and pred_I_N_3.ndim == 3
    I, N, _ = pred_I_N_3.shape

    # Calculate centroids
    cp_I_1_3 = np.mean(pred_I_N_3, axis=1, keepdims=True)
    cq_I_1_3 = np.mean(tgt_I_N_3, axis=1, keepdims=True)
    P_I_N_3  = pred_I_N_3 - cp_I_1_3
    Q_I_N_3  = tgt_I_N_3  - cq_I_1_3

    # H_i = Q_i^T P_i (align Q->P)
    H_I_3_3 = np.matmul(Q_I_N_3.swapaxes(1, 2), P_I_N_3)  # (I, 3, 3)
    
    # SVD: H = U S Vh
    U_I_3_3, _, Vh_I_3_3 = np.linalg.svd(H_I_3_3)
    
    V_I_3_3 = Vh_I_3_3.swapaxes(-2, -1)
    Ut_I_3_3 = U_I_3_3.swapaxes(-2, -1)

    # det correction: enforce det(R)=+1
    R_I_3_3 = V_I_3_3 @ Ut_I_3_3
    det_I = np.linalg.det(R_I_3_3)
    
    # Create correction matrix D
    D_I_3_3 = np.tile(np.eye(3)[np.newaxis, :, :], (I, 1, 1))
    
    # If det < 0, set bottom right element to -1
    mask_neg = det_I < 0
    if np.any(mask_neg):
        D_I_3_3[mask_neg, -1, -1] = -1.0

    # Recalculate R with correction
    R_I_3_3 = V_I_3_3 @ D_I_3_3 @ Ut_I_3_3

    # Apply rotation and translation
    aligned_Q_I_N_3 = np.matmul(Q_I_N_3, R_I_3_3) + cp_I_1_3

    # Compute per-frame RMSD
    diff_I_N_3 = aligned_Q_I_N_3 - pred_I_N_3
    rmsd_I = np.sqrt(np.mean(np.sum(diff_I_N_3 ** 2, axis=-1), axis=-1))

    return aligned_Q_I_N_3, rmsd_I


class NotConverged(Exception):
    def __init__(self, rxn_id):
        message = f'xTB calculation not converged for {rxn_id}...'
        super().__init__(message)


def build_xtb_command(xyz_file: str, charge: int, solvent: Optional[str] = None, 
                      optimize: bool = False, verbose: bool = False, 
                      gfn: int = 2, input_file: Optional[str] = None) -> List[str]:
    """
    Build XTB command with common options.
    
    Args:
        xyz_file: Path to XYZ file
        charge: Molecular charge
        solvent: Solvent name for ALPB model (optional)
        optimize: Whether to run geometry optimization
        verbose: Enable verbose output
        gfn: GFN method level (default: 2)
        input_file: Path to XTB input file for constraints (optional)
    
    Returns:
        List of command arguments
    """
    cmd = ["xtb", xyz_file]
    
    if optimize:
        cmd.append("--opt")
    
    if input_file:
        cmd.extend(["--input", input_file])
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["--chrg", str(charge), "--gfn", str(gfn)])
    
    if solvent is not None:
        cmd.extend(["--alpb", solvent])
    
    return cmd


@dataclass
class ReactionData:
    """Container for reaction information"""
    rxn_id: str
    smiles: str
    r_smiles: str
    p_smiles: str
    rxn_smiles: str

    mol: Chem.Mol
    r_mol: Chem.Mol
    p_mol: Chem.Mol

    solvent: str
    charge: int

    atoms_N: List[str]
    atoms_mn_N: List[str]
    r_atoms_N: List[str]
    p_atoms_N: List[str]

    r_mn_to_idx_dict: Dict
    p_mn_to_idx_dict: Dict
    r_idx_to_mn: np.ndarray
    p_idx_to_mn: np.ndarray
    mn_order: Dict # from mn to its order in a sorted mn list

    atoms_vdw_radii_N: np.ndarray
    atoms_vdw_radii_mn_N: np.ndarray
    atoms_cov_radii_mn_N: np.ndarray

    formed_bonds_mn_Bf: Set[Tuple[int, int]]
    broken_bonds_mn_Bf: Set[Tuple[int, int]]
    formed_bonds_idx_Bf: Set[Tuple[int, int]]
    broken_bonds_idx_Bf: Set[Tuple[int, int]]

    rxn_core_mn_R_A: List # R: num reactants, A: atom-map-numbers in rxn-core

    vdw_coef: float = 0.9  # Coefficient for target bond distance in AFIR


# Preferably use the RdKit version, since this only considers n-hop neighbors, and not other chemicaly changes such as aromaticity, formal charge, etc.1
# https://www.rdkit.org/docs/cppapi/namespaceRDKit.html#a78029a6e727e65b67e7d0798f5d3ab07
# https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/bbc534c9-c38f-96c2-f019-dfe8e24f9928@gmail.com/
def get_rxn_core_mol(rxn_smiles: str, include_n_hop_neighbor_atoms:int = 0):
    r_smiles = rxn_smiles.split(">>")[0]
    p_smiles = rxn_smiles.split(">>")[1]
    formed_bonds_mn_Bf, broken_bonds_mn_Bf = get_formed_and_broken_bonds_from_smiles(r_smiles, p_smiles)
    r_mol = Chem.MolFromSmiles(r_smiles, ps)
    p_mol = Chem.MolFromSmiles(p_smiles, ps)
    
    r_rxn_core_atom_mn_set = set()
    p_rxn_core_atom_mn_set = set()
    for changed_bonds_mn_Bf in [formed_bonds_mn_Bf, broken_bonds_mn_Bf]:
        for bond_atom_mns_2 in changed_bonds_mn_Bf:
            for atom_mn in bond_atom_mns_2:
                r_nbr_atoms_set = get_neighbors_within_n_hop_radius(r_mol, atom_mn, include_n_hop_neighbor_atoms)
                p_nbr_atoms_set = get_neighbors_within_n_hop_radius(p_mol, atom_mn, include_n_hop_neighbor_atoms)
                r_rxn_core_atom_mn_set.update(r_nbr_atoms_set)
                p_rxn_core_atom_mn_set.update(p_nbr_atoms_set)
    
    assert r_rxn_core_atom_mn_set == p_rxn_core_atom_mn_set, "Reactant and product reaction core atom sets must be equal"
    return r_rxn_core_atom_mn_set


def get_neighbors_within_n_hop_radius(mol, atom_mn, n_hop_radius):
    if n_hop_radius == 0: return {atom_mn}
    # FindAtomEnvironmentOfRadiusN returns BOND indices, so we must convert to ATOMS
    idx_to_mn = np.array([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
    mn_to_idx_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    
    bond_indices = Chem.FindAtomEnvironmentOfRadiusN(mol, n_hop_radius, mn_to_idx_dict[atom_mn], enforceSize=False)
    
    neighbor_mn = set()
    for b_idx in bond_indices:
        bond = mol.GetBondWithIdx(b_idx)
        neighbor_mn.add(int(idx_to_mn[bond.GetBeginAtomIdx()]))
        neighbor_mn.add(int(idx_to_mn[bond.GetEndAtomIdx()]))

    return neighbor_mn


def get_rxn_core_with_rdkit(rxn_smiles: str, n_hop_radius: int = 0):
    rxn = ReactionFromSmarts(rxn_smiles, useSmiles=True)
    rxn.Initialize()
    rxn_core_R_Nr = rxn.GetReactingAtoms()
    assert len(rxn_core_R_Nr) == rxn.GetNumReactantTemplates()
    
    rxn_core_mn_R_Nr = []
    for i in range(len(rxn_core_R_Nr)):
        # Get reactant i mol
        reactant_i_mol = rxn.GetReactantTemplate(i)
        
        # Convert from atom-indices to map-num. For each reactant i, the atom-indices start from 0 again.
        idx_to_mn_N = np.array([atom.GetAtomMapNum() for atom in reactant_i_mol.GetAtoms()])
        rxn_core_mn_Nr = [int(idx_to_mn_N[a_idx]) for a_idx in rxn_core_R_Nr[i]]
        
        # Get all atoms with the n-hop radius of thr rxn-core atoms
        rxn_core_with_neighbors_mn_set = set()
        for a_mn in rxn_core_mn_Nr:
            neighbors = get_neighbors_within_n_hop_radius(reactant_i_mol, a_mn, n_hop_radius)
            rxn_core_with_neighbors_mn_set.update(neighbors)
        
        # Append n-hop-rxn-core atom-map-numbers of reactant i to list
        rxn_core_mn_R_Nr.append(list(rxn_core_with_neighbors_mn_set))
    
    return rxn_core_mn_R_Nr

def get_mol_from_smiles(smiles: str) -> Chem.Mol:
    return Chem.MolFromSmiles(smiles, ps)

def get_rxn_data(rxn_id, rxn_smiles: str, solvent=None, r_or_p='r', vdw_coef=0.9) -> ReactionData:
    r_smiles = rxn_smiles.split(">>")[0]
    p_smiles = rxn_smiles.split(">>")[1]
    smiles = r_smiles if r_or_p == 'r' else p_smiles

    mol = Chem.MolFromSmiles(smiles, ps)
    r_mol = Chem.MolFromSmiles(r_smiles, ps)
    p_mol = Chem.MolFromSmiles(p_smiles, ps)

    atom_map_to_idx_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    r_mn_to_idx_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in r_mol.GetAtoms()}
    p_mn_to_idx_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in p_mol.GetAtoms()}
    r_idx_to_mn = np.array([atom.GetAtomMapNum() for atom in r_mol.GetAtoms()])
    p_idx_to_mn = np.array([atom.GetAtomMapNum() for atom in p_mol.GetAtoms()])
    
    r_idx_to_mn_sorted = np.argsort(r_idx_to_mn)
    sorted_map_numbers_N = np.sort(r_idx_to_mn)
    
    mn_order = {int(mn): i for i, mn in enumerate(sorted_map_numbers_N)}
    
    atoms_N = [atom.GetSymbol() for atom in mol.GetAtoms()]
    r_atoms_N = [atom.GetSymbol() for atom in r_mol.GetAtoms()]
    p_atoms_N = [atom.GetSymbol() for atom in p_mol.GetAtoms()]
    atoms_mn_N = [r_atoms_N[i] for i in r_idx_to_mn_sorted]
    
    charge = Chem.GetFormalCharge(mol)

    formed_bonds_mn_Bf, broken_bonds_mn_Bf = get_formed_and_broken_bonds_from_smiles(r_smiles, p_smiles)
    formed_bonds_idx_Bf = bonds_from_atom_mn_to_idx(formed_bonds_mn_Bf, atom_map_to_idx_dict)
    broken_bonds_idx_Bf = bonds_from_atom_mn_to_idx(broken_bonds_mn_Bf, atom_map_to_idx_dict)

    atoms_vdw_radii_N = np.array([qcel.vdwradii.get(atom, units='angstrom') for atom in atoms_N])
    atoms_vdw_radii_mn_N = np.array([qcel.vdwradii.get(atom, units='angstrom') for atom in atoms_mn_N])
    atoms_cov_radii_mn_N = np.array([qcel.covalentradii.get(atom, units='angstrom') for atom in atoms_mn_N])

    rxn_core_mn_R_A = get_rxn_core_with_rdkit(rxn_smiles, n_hop_radius=0)
    
    return ReactionData(
        rxn_id=int(rxn_id),
        smiles=smiles,
        r_smiles=r_smiles,
        p_smiles=p_smiles,
        rxn_smiles=rxn_smiles,
        
        mol=mol,
        r_mol=r_mol,
        p_mol=p_mol,

        solvent=solvent,
        charge=charge,
        
        atoms_N=atoms_N,
        atoms_mn_N=atoms_mn_N,
        r_atoms_N=r_atoms_N,
        p_atoms_N=p_atoms_N,

        r_mn_to_idx_dict=r_mn_to_idx_dict,
        p_mn_to_idx_dict=p_mn_to_idx_dict,
        r_idx_to_mn=r_idx_to_mn,
        p_idx_to_mn=p_idx_to_mn,
        mn_order=mn_order,
        
        atoms_vdw_radii_N=atoms_vdw_radii_N,
        atoms_vdw_radii_mn_N=atoms_vdw_radii_mn_N,
        atoms_cov_radii_mn_N=atoms_cov_radii_mn_N,
        
        formed_bonds_idx_Bf=formed_bonds_idx_Bf,
        broken_bonds_idx_Bf=broken_bonds_idx_Bf,
        formed_bonds_mn_Bf=formed_bonds_mn_Bf,
        broken_bonds_mn_Bf=broken_bonds_mn_Bf,

        rxn_core_mn_R_A=rxn_core_mn_R_A,

        vdw_coef=vdw_coef,
    )


def relax_population_with_xtb(mol_pop_P_N_3, rxn_data: ReactionData):
    mol_pop_relaxed_P_N_3 = []
    for p in range(len(mol_pop_P_N_3)):
        mol_N_3 = mol_pop_P_N_3[p]
        opt_coords_N_3 = get_xtb_relaxed_reactant_complex(mol_N_3, rxn_data)
        mol_pop_relaxed_P_N_3.append(opt_coords_N_3)
    return np.array(mol_pop_relaxed_P_N_3)

def write_xyz(atoms_N: List, coords: List, out_file="mol.xyz"):
    """ Write .xyz file """
    xyz = f"{len(atoms_N)} \n \n"
    for atomtype, coord in zip(atoms_N, coords):
        xyz += f"{atomtype}  {' '.join(list(map(str, coord)))} \n"

    with open(out_file, "w") as inp:
        inp.write(xyz)

    return out_file


def write_xyz_from_data(data: Any) -> None:
    """Write .xyz files from a PyG data object containing pos, pos_guess, and pos_gen.

    Args:
        data: PyG data object with ``atom_type``, ``pos``, ``pos_guess``, and ``pos_gen`` tensors.
    """
    
    # Convert tensors to numpy
    atom_types_N = data.atom_type.cpu().numpy()
    pos_N_3 = data.pos.cpu().numpy()
    pos_guess_N_3 = data.pos_guess.cpu().numpy()
    pos_gen_N_3 = data.pos_gen.cpu().numpy()
    
    # Convert atomic numbers to element symbols
    atoms_N = [qcel.periodictable.to_E(int(z)) for z in atom_types_N]
    
    write_xyz(atoms_N, pos_N_3, 'pos.xyz')
    write_xyz(atoms_N, pos_guess_N_3, 'pos_guess.xyz')
    write_xyz(atoms_N, pos_gen_N_3, 'pos_gen.xyz')


def write_xyz_from_tensor(atom_type, pos, out_path='pos_tens.xyz'):
    atom_types_N = atom_type.cpu().numpy()
    pos_N_3 = pos.cpu().numpy()
    atoms_N = [qcel.periodictable.to_E(int(z)) for z in atom_types_N]
    write_xyz(atoms_N, pos_N_3, out_path)


def write_pop_to_xyzs(mol_pop_P_N_3: np.ndarray, atoms_N, out_path: Path):
    os.makedirs(out_path, exist_ok=True)
    for i, mol_N_3 in enumerate(mol_pop_P_N_3):
        write_xyz(atoms_N, mol_N_3, out_path / f'mol_{i:03d}.xyz')

def get_energy(output):
    for line in output.split("\n"):
        if "TOTAL ENERGY" in line:
            return float(line.split()[3])

def get_xtb_energy_of_mol(coords_N_3: np.ndarray, rxn_data: ReactionData):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        xyz_file = tmp_path / 'mol.xyz'
        write_xyz(rxn_data.atoms_mn_N, coords_N_3, out_file=xyz_file)
        cmd = build_xtb_command('mol.xyz', rxn_data.charge, rxn_data.solvent)
        result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        return get_energy(result.stdout)

def get_xtb_energies_of_population(mol_pop_P_N_3: np.ndarray, rxn_data: ReactionData):
    HIGH_ENERGY_PENALTY = 1e10  # Extremely high energy for failed calculations
    energies_P = []
    for p in range(len(mol_pop_P_N_3)):
        mol_N_3 = mol_pop_P_N_3[p]
        energy = get_xtb_energy_of_mol(mol_N_3, rxn_data)
        if energy is None:
            energy = HIGH_ENERGY_PENALTY
        energies_P.append(energy)
    return np.array(energies_P)

def get_xtb_relaxed_reactant_complex(mol_N_3: np.ndarray, rxn_data: ReactionData):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        xyz_file = tmp_path / 'mol.xyz'
        write_xyz(rxn_data.atoms_N, mol_N_3, out_file=xyz_file)
        cmd = build_xtb_command('mol.xyz', rxn_data.charge, rxn_data.solvent, optimize=True, verbose=True)

        with open(tmp_path / 'mol.out', 'w') as out:
            process = subprocess.Popen(cmd, cwd=tmp_path, stderr=subprocess.DEVNULL, stdout=out)
            process.wait()

        opt_file = tmp_path / 'xtbopt.xyz'
        if not opt_file.exists():
            raise RuntimeError('xTB optimization did not converge.')

        # Load the optimized conformer back
        atoms = read(opt_file)
        optimized_positions = atoms.positions
        return optimized_positions


def bonds_from_atom_mn_to_idx(bonds_Bf, atom_map_to_idx_dict):
    bonds_idx = set()
    for bond in bonds_Bf:
        atom_i_mn = bond[0]
        atom_j_mn = bond[1]
        atom_i_idx = atom_map_to_idx_dict[atom_i_mn]
        atom_j_idx = atom_map_to_idx_dict[atom_j_mn]
        bonds_idx.add((atom_i_idx, atom_j_idx))
    return bonds_idx


def get_bonds(mol: Chem.Mol) -> Set[Tuple[int, int]]:
    """
    Get the bond atom-map-number pairs of a molecule.

    Args:
        mol: RDKit molecule.

    Returns:
        Set of ``(atom_map_1, atom_map_2)`` tuples with ``atom_map_1 < atom_map_2``.
    """
    bonds = set()
    for bond in mol.GetBonds():
        atom_1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
        atom_2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()

        if atom_1 < atom_2:
            bonds.add((atom_1, atom_2))
        else:
            bonds.add((atom_2, atom_1))

    return bonds

def get_formed_and_broken_bonds_from_smiles(r_smiles, p_smiles):
    """
    Identify formed bonds between reactant and product molecules.

    Returns:
    set: Formed bonds in the product.
    set: Broken bonds in the reactant.
    """
    r_mol = Chem.MolFromSmiles(r_smiles, ps)
    p_mol = Chem.MolFromSmiles(p_smiles, ps)
    
    reactant_bonds = get_bonds(r_mol)
    product_bonds = get_bonds(p_mol)

    formed_bonds = product_bonds - reactant_bonds
    broken_bonds = reactant_bonds - product_bonds

    return formed_bonds, broken_bonds


def generate_rdkit_conformers(smiles: str, n_confs: int, seed: int, useExpTorsionAnglePrefs: bool):
    conf_list_S_C_Nn_3 = []
    mol_idx_N = []
    for mol_idx, smiles_part in enumerate(smiles.split(".")):
        mol = mol_from_mapped_smiles(smiles_part)
        conf_ids = embed_conformers(
            mol,
            seed=seed,
            n_confs=n_confs,
            useExpTorsionAnglePrefs=useExpTorsionAnglePrefs
        )
        conf_C_Nn_3 = np.array([mol.GetConformer(conf_id).GetPositions() for conf_id in conf_ids])
        conf_list_S_C_Nn_3.append(conf_C_Nn_3)
        mol_idx_Nn = [mol_idx] * conf_C_Nn_3.shape[1]
        mol_idx_N.extend(mol_idx_Nn)

    conf_C_N_3 = np.concatenate(conf_list_S_C_Nn_3, axis=1)
    return conf_C_N_3, np.array(mol_idx_N)


def get_xtb_relaxed_conformer(mol: Chem.Mol, solvent: str, conf_id: int):
    """
    Optimization on RDKit especially needed when using solvents:
    https://xtb-docs.readthedocs.io/en/latest/optimization.html
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        xyz_file = tmp_path / 'conf.xyz'
        Chem.MolToXYZFile(mol, str(xyz_file), confId=conf_id)

        charge = Chem.GetFormalCharge(mol)
        cmd = build_xtb_command('conf.xyz', charge, solvent=solvent, optimize=True, verbose=True)

        with open(tmp_path / 'conf.out', 'w') as out:
            process = subprocess.Popen(cmd, cwd=tmp_path, stderr=subprocess.DEVNULL, stdout=out)
            process.wait()

        opt_file = tmp_path / 'xtbopt.xyz'
        if not opt_file.exists():
            raise RuntimeError('xTB optimization did not converge.')

        # Load the optimized conformer back
        atoms = read(opt_file)
        optimized_positions = atoms.positions
        return optimized_positions


def generate_xtb_relaxed_conformers(smiles: str, solvent: str, n_confs: int = 32, seed = None):
    conf_list_S_C_Nn_3 = []
    mol_idx_N = []
    for mol_idx, smiles_part in enumerate(smiles.split(".")):
        mol = mol_from_mapped_smiles(smiles_part)
        conf_ids = embed_conformers(
            mol,
            seed=seed,
            n_confs=n_confs,
            useExpTorsionAnglePrefs=True
        )
        conf_C_Nn_3 = np.array([get_xtb_relaxed_conformer(mol, solvent, conf_id) for conf_id in conf_ids])
        conf_list_S_C_Nn_3.append(conf_C_Nn_3)
        mol_idx_Nn = [mol_idx] * conf_C_Nn_3.shape[1]
        mol_idx_N.extend(mol_idx_Nn)

    conf_C_N_3 = np.concatenate(conf_list_S_C_Nn_3, axis=1)
    return conf_C_N_3, np.array(mol_idx_N)


def _embed_distance_geometry(
    mol: Chem.Mol,
    n_confs: int,
    seed: int,
    use_exp_torsion_angle_prefs: bool,
) -> List[int]:
    """Try standard ETKDGv3 distance-geometry embedding.

    Returns conformer IDs on success, empty list if embedding is impossible.
    A quick single-conformer probe runs first to avoid hanging on strained
    polycyclics where EmbedMultipleConfs would stall.
    """
    params = AllChem.ETKDGv3()
    params.useExpTorsionAnglePrefs = use_exp_torsion_angle_prefs
    params.useBasicKnowledge = True
    params.useRandomCoords = True
    params.randomSeed = seed
    params.maxIterations = 1000
    params.pruneRmsThresh = -1

    # Quick probe — EmbedMolecule with n=1 returns fast even for impossible cases
    probe = Chem.RWMol(mol)
    if AllChem.EmbedMolecule(probe, params) < 0:
        return []

    return list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))


def _perturb_2d_to_3d(conf: Chem.Conformer, positions_2d: list, seed: int) -> None:
    """Set conformer positions to 2D coords with random z-perturbation."""
    rng = random.Random(seed)
    for i, pos in enumerate(positions_2d):
        conf.SetAtomPosition(i, Point3D(pos.x, pos.y, rng.gauss(0, 0.5)))
    conf.Set3D(True)


def _optimize_force_field(mol: Chem.Mol, conf_id: int, max_iters: int = 2000) -> None:
    """Run UFF then MMFF force-field optimization on a single conformer."""
    # UFF — robust, handles the coarse 2D => 3D transition
    uff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    if uff is not None:
        uff.Initialize()
        uff.Minimize(maxIts=max_iters)

    # MMFF — better parameterized for organics, refines UFF result
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff_props is not None:
        mmff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
        if mmff is not None:
            mmff.Initialize()
            mmff.Minimize(maxIts=max_iters)


def _embed_via_forcefield(mol: Chem.Mol, n_confs: int, seed: int) -> List[int]:
    """Fallback embedding: 2D layout => z-perturbation => force-field optimization.

    Used for strained polycyclic molecules where distance geometry cannot
    satisfy the interatomic bounds matrix.
    """
    rdDepictor.Compute2DCoords(mol)
    ref_conf = mol.GetConformer()
    positions_2d = [ref_conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]

    conf_ids = []
    for ci in range(n_confs):
        if ci == 0:
            conf = ref_conf
        else:
            new_conf = Chem.Conformer(mol.GetNumAtoms())
            new_conf.Set3D(True)
            mol.AddConformer(new_conf, assignId=True)
            conf = mol.GetConformer(mol.GetNumConformers() - 1)

        _perturb_2d_to_3d(conf, positions_2d, seed=seed+ci)
        _optimize_force_field(mol, conf.GetId())
        conf_ids.append(conf.GetId())

    return conf_ids


def embed_conformers(
    mol: Chem.Mol,
    seed: int = 42,
    n_confs: int = 1,
    useExpTorsionAnglePrefs: bool = True,
) -> List[int]:
    """Embed 3D conformers, with automatic fallback for strained molecules.

    First attempts standard distance-geometry (ETKDGv3). If that fails (e.g.
    for strained polycyclics with contradictory distance bounds), falls back
    to 2D => force-field embedding.
    """
    conf_ids = _embed_distance_geometry(mol, n_confs, seed, useExpTorsionAnglePrefs)
    if conf_ids:
        return conf_ids

    print("  Standard embedding failed — using 2D => force-field fallback")
    return _embed_via_forcefield(mol, n_confs, seed)


def get_rdkit_reactant_conformers(rxn_data: ReactionData, n_confs: int, seed: int, save_dir: Path):
    print("Conformer generation ...")
    start_time = time.perf_counter()
    
    try:
        mol_conf_C_N_3, mol_idx_N = generate_rdkit_conformers(rxn_data.r_smiles, n_confs, seed, useExpTorsionAnglePrefs=False)
    except Exception as e:
        print(f"Error in retrieving rdkit conformers for rxn {rxn_data.rxn_id}: {e}")
        return None, None

    conf_save_dir = save_dir / 'conf'
    write_pop_to_xyzs(mol_conf_C_N_3, rxn_data.atoms_N, conf_save_dir)
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Conformer generation completed in {elapsed_time:.2f} seconds\n")

    return mol_conf_C_N_3, mol_idx_N


def mol_from_mapped_smiles(smiles: str) -> Chem.Mol:
    """Load SMILES while preserving atom mapping and explicit Hs."""
    return Chem.MolFromSmiles(smiles, ps)


def rot_x(theta):
    theta = np.asarray(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros(theta.shape + (3, 3), dtype=theta.dtype)
    R[..., 0, 0] = 1
    R[..., 1, 1] = c
    R[..., 1, 2] = -s
    R[..., 2, 1] = s
    R[..., 2, 2] = c
    return R

def rot_y(theta):
    theta = np.asarray(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros(theta.shape + (3, 3), dtype=theta.dtype)
    R[..., 0, 0] = c
    R[..., 0, 2] = s
    R[..., 1, 1] = 1
    R[..., 2, 0] = -s
    R[..., 2, 2] = c
    return R

def rot_z(theta):
    theta = np.asarray(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros(theta.shape + (3, 3), dtype=theta.dtype)
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    R[..., 2, 2] = 1
    return R

def rot_trans_mutate_population(mol_pop_P_N_3, mol_idx_N, translation_sigma, rotation_sigma):
    P = mol_pop_P_N_3.shape[0]
    M = len(set(mol_idx_N)) # number of molecules in reactant
    
    thetas_P_M_3 = np.random.normal(0.0, np.radians(rotation_sigma), size=(P, M, 3))
    Rx_P_M_3_3 = rot_x(thetas_P_M_3[..., 0])
    Ry_P_M_3_3 = rot_y(thetas_P_M_3[..., 1])
    Rz_P_M_3_3 = rot_z(thetas_P_M_3[..., 2])

    rot_P_M_3_3 = Rz_P_M_3_3 @ Ry_P_M_3_3 @ Rx_P_M_3_3
    rot_P_N_3_3 = rot_P_M_3_3[:, mol_idx_N, :, :]

    # per-(P,M) COM, then map back to atoms (P,N,3)
    mask_M_N = (np.arange(M)[:, None] == mol_idx_N[None, :])
    counts_M = mask_M_N.sum(axis=1)
    com_P_M_3 = (mol_pop_P_N_3[:, None, :, :] * mask_M_N[None, :, :, None]).sum(axis=2) / counts_M[None, :, None]
    com_P_N_3 = com_P_M_3[:, mol_idx_N, :]
    
    mol_pop_centered_P_N_3 = mol_pop_P_N_3 - com_P_N_3
    mol_pop_centered_rot_P_N_3 = np.einsum('pnij,pnj->pni', rot_P_N_3_3, mol_pop_centered_P_N_3)

    trans_vec_P_M_3 = np.random.normal(0.0, translation_sigma, size=(P, M, 3))
    trans_vec_P_N_3 = trans_vec_P_M_3[:, mol_idx_N, :]

    return mol_pop_centered_rot_P_N_3 + com_P_N_3 + trans_vec_P_N_3


def rdkit_mols_equal(m1, m2, use_chirality=True):
    """Graph isomorphism (bidirectional substructure match)."""
    if not (m1 and m2):
        return False
    if m1.GetNumAtoms() != m2.GetNumAtoms() or m1.GetNumBonds() != m2.GetNumBonds():
        return False

    return m1.HasSubstructMatch(m2, useChirality=use_chirality) and m2.HasSubstructMatch(m1, useChirality=use_chirality)


def standardized_rdkit_mol_from_smiles(smiles, use_chirality=True):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    if use_chirality:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    return mol


def get_rdkit_mol_from_xyz(xyz_N_3: np.ndarray, atoms_N, use_chirality=True, charge=0):
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
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        Chem.SanitizeMol(mol, catchErrors=True)
        if use_chirality:
            Chem.AssignStereochemistryFrom3D(mol)
        return mol
    except Exception as e:
        print(f"Error in retrieving bonds: {e}")
        return None


def swap_atom_order_from_idx_to_mn(pop_M_N_3: np.ndarray, mol: Chem.Mol):
    """
    args:
        pop_M_N_3: xyz coordinates of multiple conformers of a molecule
        mol: rdkit molecule by which the atom indices are ordered
    """
    atom_idx_to_mn_N = np.array([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
    sort_idx_by_mn_N = np.argsort(atom_idx_to_mn_N)
    return pop_M_N_3[:, sort_idx_by_mn_N, :]


def swap_atom_order_from_mn_to_idx(pop_M_N_3: np.ndarray, mol: Chem.Mol):    
    """
    args: 
        pop_M_N_3: xyz coordinates of multiple conformers of a molecule
        mol: rdkit molecule by which the atom indices are ordered
    """
    atom_idx_to_mn_N = np.array([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
    sort_idx_by_mn_N = np.argsort(atom_idx_to_mn_N)
    sort_idx_from_mn_to_idx_N = np.argsort(sort_idx_by_mn_N)
    return pop_M_N_3[:, sort_idx_from_mn_to_idx_N, :]


def p_idx_to_r_idx(p_idx: int, rxn_data: ReactionData):
    p_mn = rxn_data.p_idx_to_mn[p_idx]
    r_idx = rxn_data.r_mn_to_idx_dict[p_mn]
    return r_idx


def get_bond_forming_breaking_constrains(rxn_data: ReactionData, mol_N_3: np.ndarray = None) -> Dict:
    """
    Get a few product geometries with EA pipeline. Avg over breaking bond lengths.
    if mol_N_3 is provided use the distances from that molecule, otherwise the vdw radii sum of forming-bond atoms
    """
    bond_form_break_constraints = {}
    for bonds_mn_Bf in [rxn_data.formed_bonds_mn_Bf]:
        for a1_mn, a2_mn in bonds_mn_Bf:
            a1_mn_ord, a2_mn_ord = rxn_data.mn_order[a1_mn], rxn_data.mn_order[a2_mn]
            if mol_N_3 is None:
                dist = rxn_data.vdw_coef * (rxn_data.atoms_vdw_radii_mn_N[a1_mn_ord] + rxn_data.atoms_vdw_radii_mn_N[a2_mn_ord])
            else:
                dist = np.linalg.norm(mol_N_3[a1_mn_ord] - mol_N_3[a2_mn_ord])

            bond_form_break_constraints[(a1_mn_ord, a2_mn_ord)] = float(dist)

    return bond_form_break_constraints


def xtb_optimize_with_applied_potentials(
        rxn_data: ReactionData,
        bond_form_break_constraints: Dict,
        reactive_complex_xyz_file: Path,
        fc: float,
        output_dir: Optional[Path] = None
):
    """
    Perform XTB optimization with applied potentials in a temporary directory.

    Parameters:
    - rxn_data: Reaction data containing charge and solvent info.
    - bond_form_break_constraints: Dictionary of distance constraints.
    - reactive_complex_xyz_file (Path): Path to the reactant complex XYZ file.
    - fc (float): The force constant for the applied potentials.
    - output_dir: Directory to save the output log file. Defaults to current directory.

    Returns:
        str: Path to the XTB optimization log file.
    """
    output_dir = output_dir or Path.cwd()
    rc_filename = Path(reactive_complex_xyz_file).stem
    log_output_path = output_dir / f'{rc_filename}_path.log'

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Copy input xyz to temp directory
        shutil.copy(reactive_complex_xyz_file, tmp_path / 'mol.xyz')

        # Write constraint file in temp directory
        xtb_input_file = tmp_path / 'constraints.inp'
        with open(xtb_input_file, 'w') as f:
            f.write('$constrain\n')
            f.write(f'    force constant={fc}\n')
            for key, val in bond_form_break_constraints.items():
                f.write(f'    distance: {key[0]+1}, {key[1]+1}, {val}\n')
            f.write('$end\n')

        cmd = build_xtb_command('mol.xyz', rxn_data.charge,
                                rxn_data.solvent, optimize=True, verbose=True,
                                input_file='constraints.inp')

        with open(tmp_path / 'xtb.out', 'w') as out:
            process = subprocess.Popen(cmd, cwd=tmp_path, stderr=subprocess.DEVNULL, stdout=out)
            process.wait()

        log_file = tmp_path / 'xtbopt.log'
        if not log_file.exists():
            raise NotConverged(rxn_data.rxn_id)

        # Copy log file to output directory
        shutil.copy(log_file, log_output_path)

    return log_output_path


def read_energy_coords_file(file_path: Path) -> Tuple[list, list, list]:
    """
    Read energy and coordinate information from a file.

    Args:
        file_path: The path to the file.

    Returns:
        Tuple of ``(energies, coordinates, atom_symbols)``.
    """
    all_energies = []
    all_coords = []
    all_atoms = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            # read energy value from line starting with "energy:"
            if i + 1 >= len(lines):
                raise ValueError(f"Unexpected end of file at line {i}: expected energy line after atom count")
            if len(lines[i].split()) == 1 and lines[i+1].strip().startswith("energy:"):
                energy_line = lines[i+1].strip()
                energy_value = float(energy_line.split()[1])
                all_energies.append(energy_value)
                i += 2
            else:
                raise ValueError(f"Unexpected line format at line {i}: {lines[i]}")
            # read coordinates and symbols for next geometry
            coords = []
            atoms = []
            while i < len(lines) and len(lines[i].split()) != 1:
                atoms.append(lines[i].split()[0])
                coords.append(np.array(list(map(float,lines[i].split()[1:]))))
                i += 1

            all_coords.append(np.array(coords))
            all_atoms.append(atoms)

    return np.array(all_energies), all_coords, all_atoms


def get_rxn_core_similar_to_product_constrains(product_N_3: np.ndarray, rxn_data: ReactionData):
    constrains_C = {}
    for reactant_mn_A in rxn_data.rxn_core_mn_R_A:
        for atom_mn_2 in combinations(reactant_mn_A, 2):
            atom_1_idx = rxn_data.p_mn_to_idx_dict[atom_mn_2[0]]
            atom_2_idx = rxn_data.p_mn_to_idx_dict[atom_mn_2[1]]
            atom_idx_2 = (atom_1_idx, atom_2_idx)
            assert atom_idx_2 not in constrains_C
            constrains_C[atom_idx_2] = float(np.linalg.norm(product_N_3[atom_1_idx] - product_N_3[atom_2_idx]))

    return constrains_C


def are_new_bonds_formed(mol1_N_3: np.ndarray, mol2_N_3: np.ndarray, rxn_data: ReactionData):
    mol1 = get_rdkit_mol_from_xyz(mol1_N_3, rxn_data.atoms_mn_N, use_chirality=False, charge=rxn_data.charge)
    mol2 = get_rdkit_mol_from_xyz(mol2_N_3, rxn_data.atoms_mn_N, use_chirality=False, charge=rxn_data.charge)
    return not rdkit_mols_equal(mol1, mol2, use_chirality=False)


def relax_pop_with_constraint(rxn_data: ReactionData, mol_P_N_3: np.ndarray, path_handler: PathHandler, iter_n: int=None, use_mol_dist: bool=False, force_const: float=0.01):
    mol_relaxed_P_N_3 = []
    energies_P = []
    for mol_N_3 in mol_P_N_3:
        energies_I, mol_relaxed_I_N_3 = xtb_relax_with_fixed_changing_bond_distance(rxn_data, mol_N_3, path_handler, use_mol_dist, force_const)
        mol_relaxed_N_3 = mol_relaxed_I_N_3[-1]
        if are_new_bonds_formed(mol_N_3, mol_relaxed_N_3, rxn_data): continue
        
        energies_P.append(energies_I[-1] * HARTREE_TO_KCAL)
        mol_relaxed_P_N_3.append(mol_relaxed_N_3)

    if iter_n is not None and len(mol_relaxed_P_N_3) > 0:
        save_path_opt_relaxed = path_handler.rp_dir_struct_xyzs / f'dist-opt-relaxed-{iter_n}'
        save_path_opt_traj = path_handler.rp_dir_struct_xyzs / f'dist-opt-traj-{iter_n}'
        save_path_opt_relaxed_energies = save_path_opt_relaxed / f'energies.txt'
        write_pop_to_xyzs(mol_relaxed_P_N_3, rxn_data.atoms_mn_N, save_path_opt_relaxed)
        write_pop_to_xyzs(mol_relaxed_I_N_3, rxn_data.atoms_mn_N, save_path_opt_traj)
        save_path_opt_relaxed_energies.open('w').writelines([str(e) + '\n' for e in energies_P])

    return np.array(mol_relaxed_P_N_3), np.array(energies_P)


def xtb_relax_with_fixed_changing_bond_distance(rxn_data: ReactionData, mol_N_3, path_handler: PathHandler, use_mol_dist: bool = False, force_const: float=0.01):
    bond_forming_constraints = get_bond_forming_breaking_constrains(rxn_data, mol_N_3 = mol_N_3 if use_mol_dist else None)

    print("Bond forming constraints:", bond_forming_constraints)

    path_handler.create_ts_method_dirs()

    # Write input xyz to a temp location, will be copied to temp dir by xtb_optimize_with_applied_potentials
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
        mol_file = Path(f.name)
    write_xyz(rxn_data.atoms_mn_N, mol_N_3, out_file=mol_file)

    log_file = None
    try:
        log_file = xtb_optimize_with_applied_potentials(
            rxn_data, bond_forming_constraints, mol_file, fc=force_const,
            output_dir=path_handler.ts_temp
        )
        energies_I, all_coords_I_N_3, _ = read_energy_coords_file(log_file)
    finally:
        # Clean up temp files
        if mol_file.exists():
            mol_file.unlink()
        if log_file is not None and log_file.exists():
            log_file.unlink()

    return energies_I, all_coords_I_N_3
