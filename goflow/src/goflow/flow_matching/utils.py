from typing import List
import math
import torch
from torch import Tensor
import numpy as np
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit.Chem import AllChem
from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher, GeneticOrderMatcher, HungarianOrderMatcher, KabschMatcher
from rdkit.rdBase import BlockLogs


ps = Chem.SmilesParserParams()
ps.removeHs = False


def generate_smiles_conformers(smiles: str, n_confs: int = 1) -> np.ndarray:
    """
    Generate conformers for a given SMILES string.
    
    Args:
        smiles: SMILES string
        n_confs: Number of conformers to generate
    
    Returns:
        np.ndarray: Array of shape (N, n_confs, 3) containing conformer coordinates,
                   or None if conformer generation fails
    """
    if n_confs == 0: return None
    
    mol = Chem.MolFromSmiles(smiles, ps) 

    conf_ids = embed_conformers(mol, n_confs=n_confs)
    if conf_ids is None:
        return None
    
    # Extract coordinates for all conformers
    confs_M_N_3 = np.array([mol.GetConformer(conf_id).GetPositions() for conf_id in conf_ids])

    # 3. FORCE EXACT SIZE (Padding / Truncation)
    current_n = confs_M_N_3.shape[0]

    if current_n < n_confs:
        # PADDING: We have fewer than requested.
        # Strategy: Tile (repeat) the existing conformers to fill the gap.
        # This prevents "zero-atoms" which break message passing layers.
        n_missing = n_confs - current_n
        
        # Calculate how many full repetitions we need
        # e.g. if we have 2 and need 5 (missing 3), we need ceil(3/2)=2 tiles
        n_tiles = int(np.ceil(n_missing / current_n))
        
        # Tile and then slice off exactly what we need
        padding = np.tile(confs_M_N_3, (n_tiles, 1, 1))[:n_missing]
        
        confs_M_N_3 = np.concatenate([confs_M_N_3, padding], axis=0)
        
    elif current_n > n_confs:
        # TRUNCATION: RDKit found too many (rare, but possible)
        confs_M_N_3 = confs_M_N_3[:n_confs]

    # Sanity Check
    assert confs_M_N_3.shape[0] == n_confs, f"Shape mismatch: {confs_M_N_3.shape}"

    # Reorder to match full-molecule atom order
    full_mol = Chem.MolFromSmiles(smiles, ps)
    confs_M_N_3 = swap_atom_order_from_idx_to_mn(confs_M_N_3, full_mol)
    
    return confs_M_N_3


def embed_conformers(mol: Chem.Mol, n_confs: int = 1) -> list:
    """Embed conformers using ETKDGv3 and return list of RDKit conf IDs."""
    seed = np.random.randint(0, 2**31 - 1)
    params = AllChem.ETKDGv3()
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    params.useRandomCoords = False
    params.numThreads = 12
    params.randomSeed = seed
    params.pruneRmsThresh = -1 #0.25 if n_confs > 1 else -1

    with BlockLogs():
        for i in range(2):
            params.randomSeed = int(seed + i)
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
            if conf_ids:
                return [int(conf_id) for conf_id in conf_ids]
    
    raise RuntimeError("Could not embed conformers.")


def swap_atom_order_from_idx_to_mn(pop_M_N_3: np.ndarray, mol: Chem.Mol):
    """
    args:
        pop_M_N_3: xyz coordinates of multiple conformers of a molecule
        mol: rdkit molecule by which the atom indices are ordered
    """
    atom_idx_to_mn_N = np.array([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
    sort_idx_by_mn_N = np.argsort(atom_idx_to_mn_N)
    return pop_M_N_3[:, sort_idx_by_mn_N, :]


def write_xyz(atoms_N: List, coords: List, out_file="mol.xyz"):
    """ Write .xyz file """
    xyz = f"{len(atoms_N)} \n \n"
    for atomtype, coord in zip(atoms_N, coords):
        xyz += f"{atomtype}  {' '.join(list(map(str, coord)))} \n"

    with open(out_file, "w") as inp:
        inp.write(xyz)

    return out_file

import qcelemental as qcel
def write_xyz_from_data(data):
    """
    Write .xyz file from PyTorch tensors.
    
    Args:
        atom_types_N: Tensor of atomic numbers (integers)
        coords_N_3: Tensor of coordinates (N, 3)
        out_file: Output file path
    
    Returns:
        str: Path to output file
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


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            _, rmsd = bfm.fit(mol2)
    return rmsd


def pymatgen_rmsd(
    mol1,
    mol2,
    ignore_chirality: bool = False,
    threshold: float = 0.5,
    same_order: bool = True,
):
    rmsd = rmsd_core(mol1, mol2, threshold, same_order=same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(species=mol2.species, coords=coords)
        rmsd_reflect = rmsd_core(mol1, mol2_reflect, threshold, same_order=same_order)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd


def match_and_compute_rmsd(data):
    mol_pred = Molecule(
        species=data.atom_type.long().cpu().numpy(),
        coords=data.pos_gen.cpu().numpy(),
    )
    mol_ref = Molecule(
        species=data.atom_type.long().cpu().numpy(),
        coords=data.pos.cpu().numpy(),
    )
    try:
        rmsd = pymatgen_rmsd(
            mol_pred,
            mol_ref,
            ignore_chirality=False,
            threshold=0.5,
            same_order=True,
        )
    except Exception as e:
        print(f"Pymatgen failed with error: {e}")
        print(f"Shapes - Pred: {data.pos_gen.shape}, GT: {data.pos.shape}")
        pred_pos_N_3, gt_pos_N_3 = pred_atom_index_align(data.smiles, data.pos, data.pos_gen)
        rmsd = rmsd_loss(pred_pos_N_3, gt_pos_N_3)

    return rmsd


def compute_steric_clash_penalty(
    coords_N_3: torch.Tensor, r_threshold: float = 0.7, epsilon: float = 1.0
) -> torch.Tensor:
    """
    Compute a steric clash penalty based on a simplified LJ
    potential which only includes the repulsive 12-term. For any pair of atoms,
    if the distance r satisfies r < r_threshold then we add a penalty:

        V(r) = epsilon * [(r_threshold / r)^{12} - 1]    for r < r_threshold
        V(r) = 0                                          for r >= r_threshold

    The default r_threshold of 1.2 Å is chosen based on the observation that the
    shortest possible bond (C-C triple bond) is around this length.

    Parameters:
        coords_N_3 (torch.Tensor): Tensor of shape (N,3) with 3D positions.
        r_threshold (float): Distance threshold below which a steric clash is 
                             penalized.
        epsilon (float): Scaling factor for the penalty.

    Returns:
        torch.Tensor: The total steric clash penalty for the entire set of atoms.
    """
    # Compute all pairwise distances
    dists_N_N = torch.cdist(coords_N_3, coords_N_3, p=2)
    # Consider only unique pairs (i < j)
    mask = torch.triu(torch.ones_like(dists_N_N, dtype=torch.bool), diagonal=1)
    dists_K = dists_N_N[mask]  # K = N*(N-1)/2

    # Identify clashes where the distance is below the threshold.
    clash_mask = dists_K < r_threshold
    penalty_K = torch.zeros_like(dists_K)
    
    penalty_K[clash_mask] = epsilon * ((r_threshold / dists_K[clash_mask]) ** 12 - 1)
    total_penalty = penalty_K.sum()
    return total_penalty

def rmsd_loss(pred_N_3: Tensor, gt_N_3: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((pred_N_3 - gt_N_3) ** 2))

def kabsch_align_batched(x_0_N_3, x_1_N_3, batch):
    # x_0_N_3, x_1_N_3 are tensors of shape (N, 3)
    # batch is a 1D tensor of length N with group indices.
    device = x_0_N_3.device
    Nm = int(batch.max().item() + 1)

    # Compute counts and centers
    counts = torch.bincount(batch, minlength=Nm).to(x_0_N_3.dtype).clamp(min=1)
    
    # Compute group centroids
    centers_x0_Nm_3 = torch.zeros((Nm, 3), dtype=x_0_N_3.dtype, device=device)
    centers_x1_Nm_3 = torch.zeros((Nm, 3), dtype=x_1_N_3.dtype, device=device)
    centers_x0_Nm_3.index_add_(0, batch, x_0_N_3)
    centers_x1_Nm_3.index_add_(0, batch, x_1_N_3)
    centers_x0_Nm_3 = centers_x0_Nm_3 / counts.unsqueeze(1)
    centers_x1_Nm_3 = centers_x1_Nm_3 / counts.unsqueeze(1)

    # Center the points
    x0_centered_N_3 = x_0_N_3 - centers_x0_Nm_3[batch]
    x1_centered_N_3 = x_1_N_3 - centers_x1_Nm_3[batch]

    # Covariance Matrix construction
    prod_N_3_3 = x1_centered_N_3.unsqueeze(2) * x0_centered_N_3.unsqueeze(1)
    M_Nm_3_3 = torch.zeros((Nm, 3, 3), dtype=prod_N_3_3.dtype, device=device)
    M_Nm_3_3.index_add_(0, batch, prod_N_3_3)

    # Batched SVD
    U_Nm_3_3, _, Vt_Nm_3_3 = torch.linalg.svd(M_Nm_3_3)

    # 1. Compute determinant of the uncorrected rotation matrix UV^T
    # use the property: det(UV^T) = det(U) * det(V^T)
    R_temp = torch.bmm(U_Nm_3_3, Vt_Nm_3_3)
    det_Nm = torch.det(R_temp)
    
    # 2. Reflection Correction:
    # Instead of constructing a diagonal matrix D and doing R = U @ D @ Vt,
    # flip the sign of the last row of Vt where det < 0.
    mask_neg = det_Nm < 0
    if mask_neg.any():
        # Clone to avoid in-place modification issues if gradients are required later
        Vt_Nm_3_3 = Vt_Nm_3_3.clone() 
        Vt_Nm_3_3[mask_neg, 2, :] *= -1

    # 3. Final Rotation
    R_opt_Nm_3_3 = torch.bmm(U_Nm_3_3, Vt_Nm_3_3)

    # Apply rotation
    # (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3)
    x_1_rotated_N_3 = torch.bmm(x1_centered_N_3.unsqueeze(1), R_opt_Nm_3_3[batch]).squeeze(1)

    return x_1_rotated_N_3 + centers_x0_Nm_3[batch]


@torch.no_grad()
def get_shortest_path_x_1(
    x_target_N_3: torch.Tensor,  # e.g., x0: (N,3) — fixed target
    x_moving_N_3: torch.Tensor,  # e.g., x1: (N,3) — will be rotated/translated
    return_aligned: bool = True
):
    """
    Kabsch alignment: find R,t that minimize || (x_moving R + t) - x_target ||_F.
    Returns R (3x3), t (3,), rmsd (scalar), and optionally the aligned coords.
    """
    assert x_moving_N_3.shape == x_target_N_3.shape and x_moving_N_3.shape[-1] == 3

    # 1) center both point sets
    c_moving_1_3 = x_moving_N_3.mean(dim=0, keepdim=True)  # (1,3)
    c_target_1_3 = x_target_N_3.mean(dim=0, keepdim=True)  # (1,3)
    X = x_moving_N_3 - c_moving_1_3                          # (N,3)
    Y = x_target_N_3 - c_target_1_3                          # (N,3)

    # 2) covariance and SVD
    # Move "moving" onto "target": M = X^T Y
    M_3_3 = X.transpose(0, 1) @ Y                            # (3,3)
    U, S, Vt = torch.linalg.svd(M_3_3, full_matrices=False)  # U (3,3), Vt (3,3)

    # 3) rotation (proper, no reflection)
    V = Vt.transpose(0, 1)
    R_3_3 = V @ U.transpose(0, 1)                            # (3,3)
    if torch.det(R_3_3) < 0:
        # flip last column of V (== last row of Vt) and recompute
        V[:, -1] *= -1
        R_3_3 = V @ U.transpose(0, 1)

    # 4) translation so that (x_moving R + t) best matches x_target
    # Using row-vector convention: x' = x R + t
    t_3 = (c_target_1_3 - c_moving_1_3 @ R_3_3).squeeze(0)   # (3,)

    # 5) rmsd
    if return_aligned:
        x_aligned_N_3 = x_moving_N_3 @ R_3_3 + t_3           # (N,3)
        rmsd = torch.sqrt(torch.mean((x_aligned_N_3 - x_target_N_3) ** 2))
        #return R_3_3, t_3, rmsd, x_aligned_N_3
        return x_aligned_N_3
    else:
        # Equivalent RMSD via centered coords
        X_rot = X @ R_3_3
        rmsd = torch.sqrt(torch.mean((X_rot - Y) ** 2))
        return R_3_3, t_3, rmsd
        

def get_min_rmsd_match(matches, gt_pos, pred_pos):
    rmsd_M = []
    for match in matches:
        pred_pos_match = pred_pos[list(match)]
        gt_pos_aligned = get_shortest_path_x_1(pred_pos_match, gt_pos)
        rmsd_M.append(rmsd_loss(pred_pos_match, gt_pos_aligned))
    return list(matches[rmsd_M.index(min(rmsd_M))])


def calc_DMAE(dm_ref, dm_guess, mape=False):
    if mape:
        retval = abs(dm_ref - dm_guess) / dm_ref
    else:
        retval = abs(dm_ref - dm_guess)
    return np.triu(retval, k=1).sum() / len(dm_ref) / (len(dm_ref) - 1) * 2


def get_min_dmae_match(matches, ref_pos, prb_pos):
    dmaes = []
    for match in matches:
        match_pos = prb_pos[list(match)]
        dmae = calc_DMAE(cdist(ref_pos, ref_pos), cdist(match_pos, match_pos))
        dmaes.append(dmae)
    return list(matches[dmaes.index(min(dmaes))])


def get_substruct_matches(smarts):
    smarts_r, smarts_p = smarts.split(">>")
    mol_r = Chem.MolFromSmarts(smarts_r)
    mol_p = Chem.MolFromSmarts(smarts_p)

    matches_r = list(mol_r.GetSubstructMatches(mol_r, uniquify=False, useChirality=True))
    map_r = np.array([atom.GetAtomMapNum() for atom in mol_r.GetAtoms()]) - 1
    map_r_inv = np.argsort(map_r)
    for i in range(len(matches_r)):
        matches_r[i] = tuple(map_r[np.array(matches_r[i])[map_r_inv]])

    matches_p = list(mol_p.GetSubstructMatches(mol_p, uniquify=False, useChirality=True))
    map_p = np.array([atom.GetAtomMapNum() for atom in mol_p.GetAtoms()]) - 1
    map_p_inv = np.argsort(map_p)
    for i in range(len(matches_p)):
        matches_p[i] = tuple(map_p[np.array(matches_p[i])[map_p_inv]])

    matches = set(matches_r) & set(matches_p)
    matches = list(matches)
    matches.sort()
    return matches


def get_min_dmae_match_torch_batch(matches_M_N, pos_gt_N_3, pos_pred_S_N_3):
    """
    Given a set of matches (each a tuple of indices), ground-truth positions (pos_gt_N_3),
    and S samples of predicted positions (pos_pred_S_N_3), compute the DMAE for each match
    in each sample and return the match (as a list) with the minimal DMAE per sample.

    Args:
        matches_M_N: list or tensor of candidate matches of shape (M, N)
                     where M is the number of candidate matches and N is the number of atoms.
        pos_gt_N_3:  tensor of ground-truth atom positions of shape (N, 3).
        pos_pred_S_N_3: tensor of predicted atom positions for S samples (shape: S, N_total, 3)
                        where N_total must be large enough to index by matches_M_N.
    Returns:
        A list (or tensor) of best candidate match indices for each sample, of shape (S, N).
        Each row corresponds to the candidate match (from matches_M_N) that minimizes the DMAE
        for that sample.
    """
    matches_M_N = torch.as_tensor(matches_M_N, dtype=torch.long, device=pos_pred_S_N_3.device)

    # 1. Select and Permute Positions
    # Shape: (S, N_total, 3) -> (S, M, N, 3)
    # This gathers the specific atoms defined in matches_M_N for every sample
    candidate_pred_pos_S_M_N_3 = pos_pred_S_N_3[:, matches_M_N]

    S, M, N, _ = candidate_pred_pos_S_M_N_3.shape

    # 2. Batched Distance Matrix Calculation
    # Reshape to (S*M, N, 3) for batched cdist
    flat_pred_pos_SM_N_3 = candidate_pred_pos_S_M_N_3.reshape(S * M, N, 3)
    d_matches_SM_N_N = torch.cdist(flat_pred_pos_SM_N_3, flat_pred_pos_SM_N_3)

    # 3. Reference Distance Matrix
    # Shape: (N, N)
    d_ref_N_N = torch.cdist(pos_gt_N_3.unsqueeze(0), pos_gt_N_3.unsqueeze(0)).squeeze(0)

    # 4. Compute Absolute Difference
    # Broadcasting: (SM, N, N) - (1, N, N)
    diff_SM_N_N = torch.abs(d_matches_SM_N_N - d_ref_N_N.unsqueeze(0))

    # 5. Calculate DMAE
    # Sum over the last two dimensions (N, N)
    dmaes_SM = diff_SM_N_N.sum(dim=(-1, -2)) / (N * (N - 1))

    # 6. Find Best Match
    dmaes_S_M = dmaes_SM.view(S, M)
    best_idx_S = torch.argmin(dmaes_S_M, dim=1)

    return matches_M_N[best_idx_S]

def pred_atom_index_align(smiles, gt_atom_pos, pred_atom_pos):
    matches = get_substruct_matches(smiles)
    match = get_min_rmsd_match(matches, gt_atom_pos, pred_atom_pos)

    pred_atom_pos_match = pred_atom_pos[match]
    gt_atom_pos_aligned = get_shortest_path_x_1(pred_atom_pos_match, gt_atom_pos)

    return pred_atom_pos_match, gt_atom_pos_aligned


def pred_atom_index_align_mad(smiles, gt_atom_pos, pred_atom_pos) -> Tensor:
    matches = get_substruct_matches(smiles)
    match = get_min_dmae_match(matches, gt_atom_pos, pred_atom_pos)
    return pred_atom_pos[match]

def calc_DMAE_torch(dm_ref, dm_guess, mape=False):
    """
    Compute the Distance Matrix Absolute Error (DMAE) between two distance matrices.
    dm_ref and dm_guess are torch tensors of shape (N, N).
    """
    if mape:
        diff = torch.abs(dm_ref - dm_guess) / dm_ref
    else:
        diff = torch.abs(dm_ref - dm_guess)
    # Keep only the upper triangle (excluding the diagonal)
    diff_upper = torch.triu(diff, diagonal=1)
    N = dm_ref.shape[0]
    return 2 * diff_upper.sum() / (N * (N - 1))


def compute_chiral_volume(
    pos_N_3: torch.Tensor,
    chi_center_indices: torch.Tensor,
    rs_tags: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute the signed chiral volume for molecules.

    Used to verify if atoms define the correct stereochemistry (R vs S).
    Returns total signed chiral volume per graph.

    Args:
        pos_N_3: Positions tensor of shape (N, 3)
        chi_center_indices: Indices of 4 atoms defining each chiral center, shape (C, 4)
        rs_tags: Target chirality tags (+1 for R, -1 for S), shape (C,)
        batch: Batch indices mapping atoms to graphs, shape (N,)
        num_graphs: Number of graphs in the batch
        device: Device to use

    Returns:
        total_chiral_volume_G: Total signed chiral volume per graph, shape (G,)
    """
    total_chiral_volume_G = torch.zeros(num_graphs, device=device, dtype=pos_N_3.dtype)

    indices_C_4 = chi_center_indices.long().to(device)
    if indices_C_4.shape[0] == 0:
        return total_chiral_volume_G

    pos_indexed_C_4_3 = pos_N_3[indices_C_4]

    tetrahedral_sides_C_3_3 = pos_indexed_C_4_3[:, 1:, :] - pos_indexed_C_4_3[:, 0, :].unsqueeze(1)

    cross_product_C_3 = torch.cross(tetrahedral_sides_C_3_3[:, 0, :], tetrahedral_sides_C_3_3[:, 1, :], dim=-1)
    scalar_triple_products_C = torch.sum(tetrahedral_sides_C_3_3[:, 2, :] * cross_product_C_3, dim=1)

    rs_tag_float_C = rs_tags.float().to(device)
    aligned_volumes_C = scalar_triple_products_C * rs_tag_float_C

    mol_for_center_C = batch[indices_C_4[:, 0]]
    total_chiral_volume_G.index_add_(0, mol_for_center_C, aligned_volumes_C)

    return total_chiral_volume_G


def compute_chiral_volume_single(
    pos_N_3: torch.Tensor,
    chi_center_indices: torch.Tensor,
    rs_tags: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute the signed chiral volume for a single molecule (no batching).

    Args:
        pos_N_3: Positions tensor of shape (N, 3)
        chi_center_indices: Indices of 4 atoms defining each chiral center, shape (C, 4)
        rs_tags: Target chirality tags (+1 for R, -1 for S), shape (C,)
        device: Device to use

    Returns:
        total_chiral_volume: Total signed chiral volume (scalar)
    """
    indices_C_4 = chi_center_indices.long().to(device)
    if indices_C_4.shape[0] == 0:
        return torch.tensor(0.0, device=device, dtype=pos_N_3.dtype)

    pos_indexed_C_4_3 = pos_N_3[indices_C_4]

    tetrahedral_sides_C_3_3 = pos_indexed_C_4_3[:, 1:, :] - pos_indexed_C_4_3[:, 0, :].unsqueeze(1)

    cross_product_C_3 = torch.cross(tetrahedral_sides_C_3_3[:, 0, :], tetrahedral_sides_C_3_3[:, 1, :], dim=-1)
    scalar_triple_products_C = torch.sum(tetrahedral_sides_C_3_3[:, 2, :] * cross_product_C_3, dim=1)

    rs_tag_float_C = rs_tags.float().to(device)
    aligned_volumes_C = scalar_triple_products_C * rs_tag_float_C

    return aligned_volumes_C.sum()


def sample_harmonic_prior(edge_index: Tensor, batch: Tensor, device, dtype=torch.float32) -> Tensor:
    """
    Sample from the harmonic prior p_0(x|G) = N(0, L^+), where L = D - A is
    the graph Laplacian and L^+ is its Moore-Penrose pseudo-inverse.

    Bonded atoms are correlated in the prior: the inverse-Laplacian covariance
    constrains connected atoms to start near each other at t=0, encoding
    molecular topology into the base distribution (Jing et al., 2023).

    Args:
        edge_index: (2, E) bond connectivity (global indices in batched graph)
        batch: (N,) graph index per atom
        device: torch device
        dtype: float dtype for computation

    Returns:
        x_0: (N, 3) positions sampled from the harmonic prior
    """
    from torch_geometric.utils import to_dense_adj, to_dense_batch

    num_graphs = int(batch.max().item()) + 1
    N_total = batch.shape[0]

    # Dense adjacency (B, N_max, N_max), symmetrized and binarized
    adj = to_dense_adj(edge_index, batch=batch).to(dtype)
    adj = ((adj + adj.transpose(-1, -2)) > 0).to(dtype)

    # Graph Laplacian  L = D - A
    L = torch.diag_embed(adj.sum(dim=-1)) - adj  # (B, N_max, N_max)

    # Eigendecomposition (eigenvalues in ascending order)
    eigvals, V = torch.linalg.eigh(L)  # eigvals: (B, N_max), V: (B, N_max, N_max)

    # Pseudo-inverse sqrt: 1/sqrt(λ) for non-trivial eigenvalues, 0 otherwise.
    # Zero eigenvalues (one per connected component) span the translation modes;
    # projecting them out centers each component at the origin.
    eps = 1e-6
    inv_sqrt_eigvals = torch.where(
        eigvals > eps,
        eigvals.clamp(min=eps).rsqrt(),
        torch.zeros_like(eigvals),
    )  # (B, N_max)

    # Sample z ~ N(0, I) and compute x = V @ diag(inv_sqrt_eigvals) @ z
    N_max = L.shape[1]
    z = torch.randn(num_graphs, N_max, 3, device=device, dtype=dtype)
    x_dense = torch.bmm(V, z * inv_sqrt_eigvals.unsqueeze(-1))  # (B, N_max, 3)

    # Dense -> flat: extract real (non-padded) atoms in batch order
    _, mask = to_dense_batch(torch.ones(N_total, 1, device=device, dtype=dtype), batch)
    x_0 = x_dense[mask]  # (N_total, 3)

    return x_0
