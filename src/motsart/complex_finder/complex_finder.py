"""Main complex finder module.

Discovers optimal reactive molecular complexes from input reaction SMILES using an
evolutionary algorithm for 3D arrangement of reactants, followed by AFIR screening.
Alternatively supports neural network-based complex generation via RTSP GoFlow.
"""

from rdkit import Chem
from typing import List, Optional, Set, Tuple
import numpy as np
import time
import pandas as pd
from pathlib import Path

from hydra_zen import zen, store

from .utils import (
    ReactionData,
    rot_trans_mutate_population,
    get_rdkit_reactant_conformers,
    generate_xtb_relaxed_conformers,
    write_pop_to_xyzs,
    rdkit_mols_equal,
    standardized_rdkit_mol_from_smiles,
    get_rdkit_mol_from_xyz,
    get_rxn_data,
    relax_pop_with_constraint,
    swap_atom_order_from_idx_to_mn,
)
from .afir_path_guesser import guess_product_from_reactive_complex
from motsart.common import PathHandler
from motsart.conf_default import EnvironmentConfig, OptimizationConfig, AFIRPathGuesserParams, ALConfig


HARTREE_TO_KCAL = 627.509
KCAL_TO_HARTREE = 1 / HARTREE_TO_KCAL

ps = Chem.SmilesParserParams()
ps.removeHs = False


def get_changing_bonds_penalty(atoms_vdw_radii_N, mol_pop_P_N_3, formed_bonds_Bf, forming_bond_vdw_coef):
    penalty_total_P = np.zeros(mol_pop_P_N_3.shape[0])
    for formed_bond in formed_bonds_Bf:
        atom_i_idx = formed_bond[0]
        atom_j_idx = formed_bond[1]

        vdw_radius_sum = atoms_vdw_radii_N[atom_i_idx] + atoms_vdw_radii_N[atom_j_idx]

        atom_i_coords_P_3 = mol_pop_P_N_3[:, atom_i_idx, :]
        atom_j_coords_P_3 = mol_pop_P_N_3[:, atom_j_idx, :]

        forming_bond_len_P = np.linalg.norm(atom_i_coords_P_3 - atom_j_coords_P_3, axis=1)
        forming_bond_penalty_P = (forming_bond_len_P - forming_bond_vdw_coef*vdw_radius_sum)**2

        penalty_total_P += forming_bond_penalty_P

    return penalty_total_P


def get_steric_clash_penalty(
    atoms_vdw_radii_N: np.ndarray,
    mol_pop_P_N_3: np.ndarray,
    mol_idx_N: np.ndarray
) -> np.ndarray:
    mask_N_N = mol_idx_N[:, None] != mol_idx_N[None, :]
    i_N, j_N = np.where(mask_N_N)

    vdw_contact_K = atoms_vdw_radii_N[i_N] + atoms_vdw_radii_N[j_N]
    diff_P_K_3 = mol_pop_P_N_3[:, i_N, :] - mol_pop_P_N_3[:, j_N, :]
    r_actual_P_K = np.linalg.norm(diff_P_K_3, axis=-1)
    r_actual_P_K = np.maximum(r_actual_P_K, 1e-8)

    clash_mask_P_K = r_actual_P_K < 0.5 * vdw_contact_K[None, :]
    penalty_total_P = np.sum(clash_mask_P_K, axis=1)
    return penalty_total_P


def get_reacting_atom_mns(rxn_data: ReactionData) -> set:
    """Get atom map numbers of atoms directly involved in bond formation/breaking."""
    reacting_mns = set()
    for bond_mn in rxn_data.formed_bonds_mn_Bf | rxn_data.broken_bonds_mn_Bf:
        reacting_mns.update(bond_mn)
    return reacting_mns


def get_angle_triplets(rxn_data: ReactionData) -> List[Tuple[int, int, int]]:
    """Build angle triplets (A, B, C) where B is a reacting atom, using product topology neighbors.

    Returns list of (A_mn, B_mn, C_mn) tuples in atom map number space.
    """
    reacting_mns = get_reacting_atom_mns(rxn_data)

    # 1-hop neighbors of reacting atoms in product topology
    neighbor_mns = set()
    for atom_mn in reacting_mns:
        p_atom_idx = rxn_data.p_mn_to_idx_dict[atom_mn]
        p_atom = rxn_data.p_mol.GetAtomWithIdx(p_atom_idx)
        for nbr in p_atom.GetNeighbors():
            neighbor_mns.add(nbr.GetAtomMapNum())

    all_mns = reacting_mns | neighbor_mns

    # Triplets: B is reacting, A < C from all_mns \ {B}
    triplets = []
    for b_mn in sorted(reacting_mns):
        candidates = sorted(all_mns - {b_mn})
        for i, a_mn in enumerate(candidates):
            for c_mn in candidates[i + 1:]:
                triplets.append((a_mn, b_mn, c_mn))
    return triplets


def compute_angles_from_triplets(
    coords_P_N_3: np.ndarray,
    triplets: List[Tuple[int, int, int]],
    mn_to_idx_dict: dict,
) -> np.ndarray:
    """Compute angles at vertex B for triplets (A, B, C), vectorized over population P.

    Args:
        coords_P_N_3: (P, N, 3) atom coordinates
        triplets: list of (A_mn, B_mn, C_mn) in atom map number space
        mn_to_idx_dict: atom map number -> index in coords

    Returns:
        angles_P_T: (P, T) angles in radians
    """
    a_idx = np.array([mn_to_idx_dict[t[0]] for t in triplets])
    b_idx = np.array([mn_to_idx_dict[t[1]] for t in triplets])
    c_idx = np.array([mn_to_idx_dict[t[2]] for t in triplets])

    vec_ba = coords_P_N_3[:, a_idx, :] - coords_P_N_3[:, b_idx, :]  # (P, T, 3)
    vec_bc = coords_P_N_3[:, c_idx, :] - coords_P_N_3[:, b_idx, :]  # (P, T, 3)

    dot = np.sum(vec_ba * vec_bc, axis=-1)  # (P, T)
    norm_ba = np.linalg.norm(vec_ba, axis=-1)  # (P, T)
    norm_bc = np.linalg.norm(vec_bc, axis=-1)  # (P, T)

    cos_angle = dot / (norm_ba * norm_bc + 1e-8)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def get_product_similarity_penalty(
    mol_pop_P_N_3: np.ndarray,
    p_mol_conf_1_N_3: np.ndarray,
    rxn_data: ReactionData,
) -> np.ndarray:
    """Angle-based penalty: compares angles around reacting atoms between reactant complex and product."""
    triplets = get_angle_triplets(rxn_data)

    if len(triplets) == 0:
        return np.zeros(mol_pop_P_N_3.shape[0])

    # Reactant complex uses reactant atom ordering
    r_angles_P_T = compute_angles_from_triplets(mol_pop_P_N_3, triplets, rxn_data.r_mn_to_idx_dict)
    # Product uses product atom ordering
    p_angles_1_T = compute_angles_from_triplets(p_mol_conf_1_N_3, triplets, rxn_data.p_mn_to_idx_dict)

    angle_diff_P_T = np.abs(r_angles_P_T - p_angles_1_T)
    return np.mean(angle_diff_P_T, axis=1)

def get_reactive_complex_evolutionary_dist_based(
    rxn_data: ReactionData,
    mol_conf_C_N_3: np.ndarray,
    p_mol_conf_1_N_3: np.ndarray,
    mol_idx_N: np.ndarray,
    cfg: OptimizationConfig
):
    # Multiply population size to reach population_size
    curr_pop_size = len(mol_conf_C_N_3)
    mult_factor = cfg.dist_population_size // curr_pop_size
    mol_pop_P_N_3 = np.tile(mol_conf_C_N_3, (mult_factor, 1, 1))

    # Add initial mutations
    mol_pop_P_N_3 = rot_trans_mutate_population(mol_pop_P_N_3, mol_idx_N, cfg.dist_translation_sigma, cfg.dist_rotation_sigma)

    assert len(mol_pop_P_N_3) == cfg.dist_population_size, f"Input list length must be divisible by population_size. Got {len(mol_pop_P_N_3)} and {cfg.dist_population_size}."
    
    for g in range(cfg.dist_generations):
        # Evaluate fitness of the population
        forming_bonds_penalty_P = get_changing_bonds_penalty(rxn_data.atoms_vdw_radii_N, mol_pop_P_N_3, rxn_data.formed_bonds_idx_Bf, cfg.forming_bond_vdw_coef)
        steric_clash_penalty_P = get_steric_clash_penalty(rxn_data.atoms_vdw_radii_N, mol_pop_P_N_3, mol_idx_N)
        prod_sim_penalty_P = cfg.product_similarity_coef*get_product_similarity_penalty(mol_pop_P_N_3, p_mol_conf_1_N_3, rxn_data)
        print("forming", forming_bonds_penalty_P)
        print("steric", steric_clash_penalty_P)
        print("product-similarity penalty", np.sort(prod_sim_penalty_P))
        total_penalty_P = forming_bonds_penalty_P + steric_clash_penalty_P + prod_sim_penalty_P
        
        sorted_indices_P = np.argsort(total_penalty_P)
        elite_indices_E = sorted_indices_P[:cfg.dist_elite_num]

        if g == cfg.dist_generations - 1:
            return mol_pop_P_N_3[sorted_indices_P]

        # Tournament selection
        n_select = cfg.dist_population_size - cfg.dist_elite_num
        tournament_k = 5
        candidates_NE_3 = np.random.randint(0, cfg.dist_population_size, size=(n_select, tournament_k))
        tournament_penalties_NE_3 = total_penalty_P[candidates_NE_3]
        winners_N = np.argmin(tournament_penalties_NE_3, axis=1)
        selected_indices_NE = candidates_NE_3[np.arange(n_select), winners_N]

        # Selection step
        new_pop_indices_P = np.concatenate((elite_indices_E, selected_indices_NE))
        mol_pop_P_N_3 = mol_pop_P_N_3[new_pop_indices_P]

        # Mutate non-elite individuals (sigma annealing)
        progress = g / max(cfg.dist_generations - 1, 1)
        sigma_trans = cfg.dist_translation_sigma * (1.0 - 0.8 * progress)
        sigma_rot = cfg.dist_rotation_sigma * (1.0 - 0.8 * progress)
        mol_pop_P_N_3[cfg.dist_elite_num:] = rot_trans_mutate_population(mol_pop_P_N_3[cfg.dist_elite_num:], mol_idx_N, sigma_trans, sigma_rot)
        mol_pop_P_N_3 = mol_pop_P_N_3 - np.mean(mol_pop_P_N_3, axis=1, keepdims=True)


def get_dihedral_indices(mol, atom_subset_indices):
    """Finds all paths of length 4 (torsions) fully contained within the subset."""
    subset_set = set(atom_subset_indices)
    dihedral_indices = []
    
    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        
        if j not in subset_set or k not in subset_set: continue
            
        j_neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k]
        k_neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j]
        
        for i in j_neighbors:
            if i in subset_set:
                for l in k_neighbors:
                    if l in subset_set:
                        dihedral_indices.append([i, j, k, l])
                
    return np.array(dihedral_indices)

def compute_dihedrals(coords, indices):
    """Vectorized calculation of dihedral angles (degrees)."""
    p0 = coords[:, indices[:, 0], :]
    p1 = coords[:, indices[:, 1], :]
    p2 = coords[:, indices[:, 2], :]
    p3 = coords[:, indices[:, 3], :]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    
    # Normalize b1
    b1 /= np.linalg.norm(b1, axis=2, keepdims=True)

    v = b0 - np.sum(b0 * b1, axis=2, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=2, keepdims=True) * b1

    x = np.sum(v * w, axis=2)
    y = np.sum(np.cross(b1, v) * w, axis=2)
    
    return np.degrees(np.arctan2(y, x))

def calc_core_torsion_deviation(
    r_coords_N_3: np.ndarray, 
    p_coords_1_3: np.ndarray, 
    r_mol: Chem.Mol, 
    r_core_idx_A: np.ndarray, 
    p_core_idx_A: np.ndarray
) -> np.ndarray:
    """
    Computes the mean absolute torsion deviation between reactant conformers 
    and a product reference for a specific reaction core.
    """
    # 1. Identify valid torsion chains (indices) within the reactant core subset
    torsion_indices_N_4 = get_dihedral_indices(r_mol, r_core_idx_A)
    
    if len(torsion_indices_N_4) == 0:
        return np.zeros(len(r_coords_N_3))

    # 2. Compute angles for all reactant conformers (N_confs, N_torsions)
    r_angles = compute_dihedrals(r_coords_N_3, torsion_indices_N_4)
    
    # 3. Map Reactant torsion indices to Product torsion indices
    #    Create a lookup: Reactant_Global_Idx -> Product_Global_Idx
    r_to_p_map = dict(zip(r_core_idx_A, p_core_idx_A))
    torsion_indices_N_4 = np.vectorize(r_to_p_map.get)(torsion_indices_N_4)
    
    # 4. Compute angles for the product reference (1, N_torsions)
    p_angles = compute_dihedrals(p_coords_1_3, torsion_indices_N_4) 
    
    # 5. Compute Circular Difference: min(|a-b|, 360 - |a-b|)
    abs_diff = np.abs(r_angles - p_angles)
    angle_err = np.minimum(abs_diff, 360.0 - abs_diff)
    
    # Return mean angular deviation per conformer
    return np.mean(angle_err, axis=1)


def filter_reactant_conformers_similar_to_product_rxn_core(
    r_C_N_3: np.ndarray, 
    p_1_N_3: np.ndarray,
    rxn_data: ReactionData,
    n_confs: int,
):
    # For each reactant, take rxn_core and compute difference to product for all conformers (kabsch aligned)
    err_C = np.zeros(len(r_C_N_3))
    for reactant_i_rxn_core_mn_A in rxn_data.rxn_core_mn_R_A:
        r_core_idx_A = np.array([rxn_data.r_mn_to_idx_dict[mn] for mn in reactant_i_rxn_core_mn_A])
        p_core_idx_A = np.array([rxn_data.p_mn_to_idx_dict[mn] for mn in reactant_i_rxn_core_mn_A])
        err_C += calc_core_torsion_deviation(r_C_N_3, p_1_N_3, rxn_data.r_mol, r_core_idx_A, p_core_idx_A)
    
    best_indices_C = np.argsort(err_C)
    r_Cf_N_3 = r_C_N_3[best_indices_C[:n_confs]]
    print(f"Product similarity filter errs for rxn {rxn_data.rxn_id}:", err_C[best_indices_C[:2]], "...", err_C[best_indices_C[n_confs-2:n_confs]])
    
    return r_Cf_N_3


def one_evolutionary_optimization_round(
    rxn_data: ReactionData,
    cfg: OptimizationConfig,
    seed: int,
    path_handler: PathHandler,
    iter_n: int,
):
    start_time_total = time.perf_counter()
    
    # ------------- Conformer Generation -------------
    rp_dir_struct_xyzs_iter_n = path_handler.get_iter_n_dir(iter_n)
    mol_conf_C_N_3, mol_idx_N = get_rdkit_reactant_conformers(rxn_data, cfg.n_confs, seed, save_dir=rp_dir_struct_xyzs_iter_n)
    if mol_conf_C_N_3 is None:
        return None, None
    p_mol_conf_1_N_3, _ = generate_xtb_relaxed_conformers(rxn_data.p_smiles, rxn_data.solvent, n_confs=1, seed=seed)

    mol_conf_Cf_N_3 = filter_reactant_conformers_similar_to_product_rxn_core(mol_conf_C_N_3, p_mol_conf_1_N_3, rxn_data, cfg.n_confs_after_product_similarity_filter)
    write_pop_to_xyzs(mol_conf_Cf_N_3[:,mol_idx_N==0,:], np.array(rxn_data.atoms_N)[mol_idx_N==0], path_handler.rp_dir / f'conf_similar_to_p')

    # ------------- Distance Optimization -------------
    mol_pop_P_N_3 = dist_optimization(mol_conf_Cf_N_3, p_mol_conf_1_N_3, mol_idx_N, rxn_data, cfg, rp_dir_struct_xyzs_iter_n)

    # ------------- Switch from atom-idx order to atom-map-number -------------
    mol_pop_P_N_3 = swap_atom_order_from_idx_to_mn(mol_pop_P_N_3, rxn_data.r_mol)
    
    # ------------- Relaxation -------------
    st = time.perf_counter()
    mol_pop_Pf_N_3, energies_Pf = relax_pop_with_constraint(rxn_data, mol_pop_P_N_3[:cfg.n_rcs_to_screen_for_energy], path_handler, iter_n)
    if len(energies_Pf) == 0:
        print(f"Empty population for rxn {rxn_data.rxn_id}")
        return None, None
    print(f"Relaxed population energies of rxn {rxn_data.rxn_id}: {np.sort(energies_Pf)}, in {time.perf_counter()-st:.2f} seconds")
    
    print(f"TOTAL TIME, one optimization round: {time.perf_counter()-start_time_total:.2f} seconds")
    best_idx = np.argmin(energies_Pf)
    return mol_pop_Pf_N_3[best_idx], energies_Pf[best_idx]
    

def dist_optimization(
        mol_conf_C_N_3: np.ndarray,
        p_mol_conf_1_N_3: np.ndarray,
        mol_idx_N: np.ndarray, 
        rxn_data: ReactionData, 
        cfg: OptimizationConfig, 
        save_path: Path,
):
    """
    1. Perform evoluationary distance optimization.
    2. Relax those reactant complexes with changing-bond distance constraint
    """
    print("Distance optimization ...")
    start_time = time.perf_counter()
    
    pop_opt_P_N_3 = get_reactive_complex_evolutionary_dist_based(
        rxn_data,
        mol_conf_C_N_3,
        p_mol_conf_1_N_3,
        mol_idx_N,
        cfg,
    )
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Distance optimization completed in {elapsed_time:.2f} seconds\n")

    save_path_opt = save_path / 'dist-opt'
    write_pop_to_xyzs(pop_opt_P_N_3[:8], rxn_data.atoms_N, save_path_opt)

    return pop_opt_P_N_3


def filter_mols_with_same_struct_as_smiles(mol_pop_P_N_3: np.ndarray, rxn_data):
    ref_mol = standardized_rdkit_mol_from_smiles(rxn_data.smiles)
    
    mol_pop_filter_P = []
    for mol_N_3 in mol_pop_P_N_3:
        mol_relaxed = get_rdkit_mol_from_xyz(mol_N_3, rxn_data.atoms_N, charge=rxn_data.charge)
        mol_pop_filter_P.append(rdkit_mols_equal(mol_relaxed, ref_mol))
    
    return np.array(mol_pop_filter_P)


def get_highest_reward_members_within_best_kcal_range(mol_pop_relaxed_P_N_3, energies_relaxed_P, reward_relaxed_P, rxn_data: ReactionData, kcal_range=1.0, n_members=4):
    hartree_range = KCAL_TO_HARTREE * kcal_range

    # remove pop-members where reactant-complex smiles is not sames as original smiles
    mol_pop_filter_P = filter_mols_with_same_struct_as_smiles(mol_pop_relaxed_P_N_3, rxn_data)
    if np.all(~mol_pop_filter_P):
        return None, None, None
    mol_pop_relaxed_P_N_3 = mol_pop_relaxed_P_N_3[mol_pop_filter_P]
    energies_relaxed_P = energies_relaxed_P[mol_pop_filter_P]
    reward_relaxed_P = reward_relaxed_P[mol_pop_filter_P]

    lowest_energy = np.min(energies_relaxed_P)
    best_pop_members_P = (energies_relaxed_P - lowest_energy) < hartree_range
    best_pop_Pb_N_3 = mol_pop_relaxed_P_N_3[best_pop_members_P]
    best_pop_rewards_Pb = reward_relaxed_P[best_pop_members_P]
    best_pop_energies_Pb = energies_relaxed_P[best_pop_members_P]
    
    best_members_w_highest_reward_M = np.argsort(-best_pop_rewards_Pb)[:n_members]
    best_members_highest_reward_M = best_pop_rewards_Pb[best_members_w_highest_reward_M]
    best_members_w_highest_reward_energy_M = best_pop_energies_Pb[best_members_w_highest_reward_M]
    best_members_M_N_3 = best_pop_Pb_N_3[best_members_w_highest_reward_M]

    return best_members_M_N_3, best_members_w_highest_reward_energy_M, best_members_highest_reward_M


def multiple_evolutionary_optimization_rounds(
        rxn_data: ReactionData,
        path_handler: PathHandler,
        cfg: OptimizationConfig,
    ):
    
    # ------------------ Preliminaries -------------------------    
    print(f"\n##################### Processing rxn {rxn_data.rxn_id} started #####################")
    
    results = {'energy': []}
    best_members_I_N_3 = []
    for n in range(cfg.n_EA_rounds):
        print(f"\nRxn {rxn_data.rxn_id}, iter {n} ----------------------")
        
        best_member_N_3, best_member_energy = one_evolutionary_optimization_round(
            rxn_data=rxn_data,
            cfg=cfg,
            seed=cfg.seed+n,
            path_handler=path_handler,
            iter_n=n
        )
        if best_member_N_3 is None:
            print(f"Skipping EA round {n} for rxn {rxn_data.rxn_id}")
            continue
        
        best_members_I_N_3.append(best_member_N_3)
        results['energy'].append(best_member_energy)
    
    best_members_I_N_3 = np.array(best_members_I_N_3)
    
    if len(best_members_I_N_3) == 0:
        print(f"!!!REACTION {rxn_data.rxn_id} COULD NOT BE PROCESSED!!!")
    
    write_pop_to_xyzs(best_members_I_N_3, rxn_data.atoms_mn_N, path_handler.rp_dir_final)


def complex_finder_task(
        optim_cfg: OptimizationConfig,
        afir_cfg: AFIRPathGuesserParams,
        env: EnvironmentConfig,
        multihead_module=None,
        al_cfg: ALConfig = None,
):
    df_smi = pd.read_csv(env.rxn_csv, sep=',', header=None)
    if env.rxn_id is not None:
        rxn_id = env.rxn_id
        matching_rows = df_smi[df_smi[0] == rxn_id]
        rxn_smiles = matching_rows[1].values[0]
    else:
        rxn_id = df_smi[0].values[env.rxn_num]
        rxn_smiles = df_smi[1].values[env.rxn_num]

    if optim_cfg.complex_method == "rtsp_goflow":
        # Use RTSP Guesser to generate R, P, and TS geometries via neural network
        _run_rtsp_guesser(rxn_id, rxn_smiles, env, multihead_module, al_cfg)
    else:
        # Default: evolutionary optimization + AFIR
        _run_complex_finder(rxn_id, rxn_smiles, env, optim_cfg, afir_cfg)


def _run_rtsp_guesser(rxn_id, rxn_smiles, env: EnvironmentConfig, multihead_module, al_cfg: ALConfig):
    """Generate R, P, and TS geometries using RTSP Guesser (MultiHead GoFlow)."""
    from motsart.learning.rtsp_guesser import RTSPGuesser

    if multihead_module is None:
        raise ValueError("multihead_module is required when using complex_method='rtsp_goflow'")
    if al_cfg is None:
        raise ValueError("al_cfg is required when using complex_method='rtsp_goflow'")

    guesser = RTSPGuesser(
        multihead_module=multihead_module,
        rxn_id=str(rxn_id),
        rxn_smiles=rxn_smiles,
        results_folder=env.results_folder,
        n_conformers=al_cfg.n_conformers,
        num_samples=al_cfg.num_samples,
    )

    st = time.time()
    guesser.run_inference()
    print(f"RTSP inference completed in {time.time() - st:.2f}s")


def _run_complex_finder(rxn_id, rxn_smiles, env: EnvironmentConfig, optim_cfg: OptimizationConfig, afir_cfg: AFIRPathGuesserParams):
    """Default complex finding: evolutionary optimization + AFIR product guessing."""
    path_handler = PathHandler(rxn_id, 'r', 'afir', results_folder=env.results_folder)

    rxn_data_r = get_rxn_data(rxn_id, rxn_smiles, solvent=env.solvent, r_or_p='r', vdw_coef=env.vdw_coef)
    rxn_data_p = get_rxn_data(rxn_id, rxn_smiles, solvent=env.solvent, r_or_p='p', vdw_coef=env.vdw_coef)

    # -------------- Find reactant complexes --------------
    path_handler.rm_existing_rp_dir()
    path_handler.chdir_to_temp()
    multiple_evolutionary_optimization_rounds(rxn_data_r, path_handler, optim_cfg)
    path_handler.chdir_to_proj_root()

    # -------------- AFIR product guesser --------------
    path_handler.rm_existing_ts_dir()
    path_handler.chrdir_to_ts_temp()
    guess_product_from_reactive_complex(rxn_data_p, path_handler, afir_cfg)
    path_handler.chdir_to_proj_root()



if __name__ == "__main__":
    import motsart.complex_finder.conf
    import motsart.learning.conf
    store(
        complex_finder_task,
        name="complex_finder_root",
        hydra_defaults=[
            "_self_",
            {"optim_cfg": "test"},
            {"afir_cfg": "test"},
            {"env": "test"},
            {"multihead_module": "mhfm_default"},
            {"al_cfg": "test"},
        ]
    )
    store.add_to_hydra_store()
    zen(complex_finder_task).hydra_main(
        config_name="complex_finder_root",
        version_base="1.3"
    )