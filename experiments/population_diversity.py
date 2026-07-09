#!/usr/bin/env python3
"""Quantify and visualize how ComplexFinder's population diversity evolves over
the evolutionary-search generations.

Diversity metrics (logged per generation)
------------------------------------------
* ``mean_pairwise_rmsd``  -- mean pairwise heavy-atom RMSD (A), Kabsch-aligned to
  the current best member. The standard geometric ensemble-diversity measure; a
  sustained non-zero plateau is the visual proof that diversity does not collapse.
* ``n_clusters``          -- number of distinct structural clusters (greedy leader
  clustering at ``--rmsd-cutoff`` A). An intuitive "effective number of surviving
  solutions"; collapses to ~1 under greedy selection.
* ``fb_dist_std``         -- spread (A) of the forming-bond distances across the
  population: diversity in the chemically meaningful coordinate the EA optimizes.
* ``angle_std``           -- spread (deg) of the reacting-atom approach angles.
* ``fitness_std`` / ``fitness_min`` -- penalty spread and best penalty; the latter
  shows both selection schemes make comparable optimization progress.

Usage
-----
    python experiments/population_diversity.py
    python experiments/population_diversity.py --n-rxns 8 --generations 300
    python experiments/population_diversity.py --rxn-ids 0,3,7,12 --selections tournament

Run from the repo root (the default ``--rxn-csv`` is repo-relative).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from motsart.conf_default import OptimizationConfig
from motsart.complex_finder.utils import (
    get_rxn_data,
    get_rdkit_reactant_conformers,
    generate_xtb_relaxed_conformers,
)
from motsart.complex_finder.complex_finder import (
    get_reactive_complex_evolutionary_dist_based,
    filter_reactant_conformers_similar_to_product_rxn_core,
    get_angle_triplets,
    compute_angles_from_triplets,
)


# ----------------------------------------------------------------------------
# Diversity metrics
# ----------------------------------------------------------------------------

def _kabsch_align_to_ref(members_S_K_3: np.ndarray, ref_K_3: np.ndarray) -> np.ndarray:
    """Kabsch-align every member onto a common reference (vectorized over members).

    Returns the aligned members in the reference frame, shape ``(S, K, 3)``.
    Mirrors ``motsart.complex_finder.utils.kabsch_align_pairwise_I`` but aligns a
    whole batch onto one reference and is silent (no prints).
    """
    pred = np.broadcast_to(ref_K_3, members_S_K_3.shape)            # (S, K, 3)
    tgt = members_S_K_3
    cp = pred.mean(axis=1, keepdims=True)
    cq = tgt.mean(axis=1, keepdims=True)
    P = pred - cp
    Q = tgt - cq
    H = np.matmul(Q.swapaxes(1, 2), P)                             # (S, 3, 3)
    U, _, Vh = np.linalg.svd(H)
    V = Vh.swapaxes(-2, -1)
    Ut = U.swapaxes(-2, -1)
    R = V @ Ut
    det = np.linalg.det(R)
    D = np.broadcast_to(np.eye(3), R.shape).copy()
    D[det < 0, -1, -1] = -1.0
    R = V @ D @ Ut
    return np.matmul(Q, R) + cp                                    # aligned onto pred


def _leader_clusters(rmsd_mat: np.ndarray, tau: float) -> int:
    """Greedy leader (Taylor-Butina-style) clustering; returns the cluster count."""
    n = rmsd_mat.shape[0]
    assigned = np.zeros(n, dtype=bool)
    n_clusters = 0
    for i in range(n):
        if assigned[i]:
            continue
        n_clusters += 1
        assigned |= rmsd_mat[i] <= tau
    return n_clusters


class DiversityRecorder:
    """Per-generation diversity callback for ``get_reactive_complex_evolutionary_dist_based``.

    Captures the population state at every generation (before selection). The EA
    population is in atom-*index* order, so reaction-core descriptors use
    ``rxn_data.r_mn_to_idx_dict`` exactly like the production penalty terms.
    """

    def __init__(self, rxn_data, subsample: int, rmsd_cutoff: float, seed: int):
        self.rxn = rxn_data
        self.subsample = subsample
        self.tau = rmsd_cutoff
        self.rng = np.random.default_rng(seed)
        self.records: list[dict] = []

        self.heavy_mask = np.array([s != "H" for s in rxn_data.atoms_N])
        fb = np.array(sorted(rxn_data.formed_bonds_idx_Bf), dtype=int)
        self.fb_pairs = fb if fb.size else None                    # (B, 2) or None
        self.triplets = get_angle_triplets(rxn_data)

    def __call__(self, generation: int, pop_P_N_3: np.ndarray, penalty_P: np.ndarray):
        P = pop_P_N_3.shape[0]

        # --- geometric diversity on a heavy-atom subsample, aligned to the best member ---
        S = min(self.subsample, P)
        idx_S = self.rng.choice(P, size=S, replace=False) if S < P else np.arange(P)
        best = int(np.argmin(penalty_P))
        heavy = pop_P_N_3[:, self.heavy_mask, :]
        ref_K_3 = heavy[best]
        aligned = _kabsch_align_to_ref(heavy[idx_S], ref_K_3)      # (S, K, 3)
        K = aligned.shape[1]

        flat = aligned.reshape(S, -1)
        pair_rmsd = pdist(flat) / np.sqrt(K)                       # Euclidean(3K) -> RMSD
        mean_pairwise_rmsd = float(pair_rmsd.mean()) if pair_rmsd.size else 0.0
        n_clusters = _leader_clusters(squareform(pair_rmsd), self.tau)
        rmsd_to_best = float(
            np.sqrt(np.mean(np.sum((aligned - ref_K_3) ** 2, axis=-1), axis=-1)).mean()
        )

        # --- chemical-coordinate diversity on the full population ---
        if self.fb_pairs is not None:
            d_PB = np.linalg.norm(
                pop_P_N_3[:, self.fb_pairs[:, 0], :] - pop_P_N_3[:, self.fb_pairs[:, 1], :],
                axis=-1,
            )
            fb_dist_std = float(d_PB.std(axis=0).mean())
        else:
            fb_dist_std = np.nan

        if self.triplets:
            ang = np.degrees(
                compute_angles_from_triplets(pop_P_N_3, self.triplets, self.rxn.r_mn_to_idx_dict)
            )
            angle_std = float(ang.std(axis=0).mean())
        else:
            angle_std = np.nan

        self.records.append(dict(
            generation=generation,
            mean_pairwise_rmsd=mean_pairwise_rmsd,
            rmsd_to_best=rmsd_to_best,
            n_clusters=n_clusters,
            fb_dist_std=fb_dist_std,
            angle_std=angle_std,
            fitness_std=float(penalty_P.std()),
            fitness_min=float(penalty_P.min()),
        ))


# ----------------------------------------------------------------------------
# Single reaction / single selection-scheme run
# ----------------------------------------------------------------------------

def run_one(rxn_id, rxn_smiles, solvent, selection, cfg, args, tmp_dir: Path):
    """Run conformer generation + EA for one reaction and one selection scheme.

    Conformers and the product reference are generated by the caller and reused
    across selection schemes, so only the EA differs between schemes. Returns the
    per-generation records, or ``None`` on conformer-generation failure.
    """
    rxn_data = run_one._cache[rxn_id]["rxn_data"]
    confs_kept = run_one._cache[rxn_id]["confs_kept"]
    p_ref_1_N_3 = run_one._cache[rxn_id]["p_ref"]
    mol_idx_N = run_one._cache[rxn_id]["mol_idx_N"]

    # Adapt the population size to the number of kept conformers so it stays an
    # exact multiple (the EA asserts divisibility).
    kept = len(confs_kept)
    mult = max(1, args.target_population // kept)
    cfg.dist_population_size = mult * kept

    recorder = DiversityRecorder(rxn_data, args.subsample, args.rmsd_cutoff, seed=args.sample_seed)

    # Seed numpy so the initial tiling+mutation is identical across selection
    # schemes; only the per-generation selection then differs.
    np.random.seed(args.ea_seed)
    get_reactive_complex_evolutionary_dist_based(
        rxn_data, confs_kept, p_ref_1_N_3, mol_idx_N, cfg,
        selection=selection,
        gen_callback=recorder,
        truncation_frac=args.truncation_frac,
    )

    df = pd.DataFrame(recorder.records)
    df.insert(0, "selection", selection)
    df.insert(0, "rxn_id", rxn_id)
    return df


run_one._cache = {}


def prepare_reaction(rxn_id, rxn_smiles, solvent, args, tmp_dir: Path) -> bool:
    """Generate (and cache) RDKit conformers + xTB product reference for a reaction."""
    rxn_data = get_rxn_data(rxn_id, rxn_smiles, solvent=solvent, r_or_p="r")

    confs_C_N_3, mol_idx_N = get_rdkit_reactant_conformers(
        rxn_data, args.n_confs, seed=args.ea_seed, save_dir=tmp_dir / f"rxn_{rxn_id}"
    )
    if confs_C_N_3 is None:
        print(f"[skip] rxn {rxn_id}: conformer generation failed")
        return False

    p_ref_1_N_3, _ = generate_xtb_relaxed_conformers(
        rxn_data.p_smiles, rxn_data.solvent, n_confs=1, seed=args.ea_seed
    )

    confs_kept = filter_reactant_conformers_similar_to_product_rxn_core(
        confs_C_N_3, p_ref_1_N_3, rxn_data, args.kept
    )

    run_one._cache[rxn_id] = dict(
        rxn_data=rxn_data, confs_kept=confs_kept, p_ref=p_ref_1_N_3, mol_idx_N=mol_idx_N,
    )
    return True


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

# Okabe-Ito colorblind-safe palette: blue + vermillion.
_METHOD_STYLE = {
    "tournament": ("#0072B2", "Tournament"),
    "truncation": ("#D55E00", "Greedy"),
    "greedy": ("#D55E00", "Greedy"),
}


def _band(ax, df, metric, methods, *, ylabel, title, per_rxn=True):
    for method in methods:
        if method not in _METHOD_STYLE:
            continue
        color, label = _METHOD_STYLE[method]
        sub = df[df.selection == method]
        if sub.empty:
            continue
        if per_rxn:
            for _, g in sub.groupby("rxn_id"):
                g = g.sort_values("generation")
                ax.plot(g.generation, g[metric], color=color, alpha=0.13, lw=0.8)
        agg = (sub.groupby("generation")[metric]
                  .agg(med="median",
                       q1=lambda s: s.quantile(0.25),
                       q3=lambda s: s.quantile(0.75))
                  .reset_index())
        ax.plot(agg.generation, agg.med, color=color, lw=2.2, label=label)
        ax.fill_between(agg.generation, agg.q1, agg.q3, color=color, alpha=0.20, lw=0)
    ax.set_xlabel("Evolutionary generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)


def make_figures(df: pd.DataFrame, out_dir: Path, n_rxns: int, rmsd_cutoff: float, subsample: int):
    methods = [m for m in ("tournament", "truncation", "greedy") if m in df.selection.unique()]
    df = df.copy()
    # Fraction of the subsampled population that is structurally distinct (<= 1.0).
    df["n_clusters_frac"] = df["n_clusters"] / subsample

    # --- Figure: cluster diversity fraction ---
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _band(ax, df, "n_clusters_frac", methods,
          ylabel="Distinct cluster fraction",
          title="")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"diversity_fig.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Detailed absolute-value panels ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _band(axes[0, 0], df, "mean_pairwise_rmsd", methods,
          ylabel="Mean pairwise RMSD (Å)", title="Geometric diversity (absolute)")
    _band(axes[0, 1], df, "n_clusters", methods,
          ylabel="Number of clusters", title="Effective diversity (absolute)")
    _band(axes[0, 2], df, "rmsd_to_best", methods,
          ylabel="Mean RMSD to best (Å)", title="Spread around the elite")
    _band(axes[1, 0], df, "fb_dist_std", methods,
          ylabel="Std of forming-bond dist. (Å)", title="Coordinate diversity: bond distances")
    _band(axes[1, 1], df, "angle_std", methods,
          ylabel="Std of approach angle (deg)", title="Coordinate diversity: approach angles")
    _band(axes[1, 2], df, "fitness_min", methods,
          ylabel="Best penalty (a.u.)", title="Optimization progress (lower = better)")
    fig.suptitle(f"ComplexFinder diversity & convergence diagnostics "
                 f"(median ± IQR over {n_rxns} reactions)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"diversity_metrics.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def select_reactions(df_csv: pd.DataFrame, args) -> list:
    if args.rxn_ids:
        wanted = [int(x) for x in args.rxn_ids.split(",")]
        return [r for r in wanted if r in set(df_csv[0].values)]
    ids = df_csv[0].values
    rng = np.random.default_rng(args.sample_seed)
    return sorted(int(x) for x in rng.choice(ids, size=min(args.n_rxns, len(ids)), replace=False))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rxn-csv", default="data/cyclo32_atom_mapped_small.csv")
    p.add_argument("--n-rxns", type=int, default=8)
    p.add_argument("--rxn-ids", default=None, help="Comma-separated ids (overrides random sampling).")
    p.add_argument("--sample-seed", type=int, default=0, help="Seed for reaction sampling & metric subsampling.")
    p.add_argument("--ea-seed", type=int, default=42, help="Seed for conformers and the EA (shared across schemes).")
    p.add_argument("--selections", default="tournament,truncation",
                   help="Comma-separated selection schemes to compare.")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--n-confs", type=int, default=128, help="RDKit conformers generated per reactant set.")
    p.add_argument("--kept", type=int, default=64, help="Conformers kept after product-similarity filtering.")
    p.add_argument("--target-population", type=int, default=512)
    p.add_argument("--elite", type=int, default=10)
    p.add_argument("--translation-sigma", type=float, default=0.5)
    p.add_argument("--rotation-sigma", type=float, default=25.0)
    p.add_argument("--forming-bond-vdw-coef", type=float, default=1.25)
    p.add_argument("--product-similarity-coef", type=float, default=10.0)
    p.add_argument("--truncation-frac", type=float, default=0.1)
    p.add_argument("--subsample", type=int, default=64, help="Members sampled per generation for RMSD/clustering.")
    p.add_argument("--rmsd-cutoff", type=float, default=1.0, help="Leader-clustering RMSD cutoff (A).")
    p.add_argument("--solvent", default=None, help="Override solvent (default: read CSV column 3, else 'water').")
    p.add_argument("--out-dir", default="results_diversity")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_csv = pd.read_csv(args.rxn_csv, sep=",", header=None)
    rxn_ids = select_reactions(df_csv, args)
    selections = [s.strip() for s in args.selections.split(",") if s.strip()]
    print(f"Reactions: {rxn_ids}")
    print(f"Selection schemes: {selections}")

    cfg = OptimizationConfig(
        n_confs=args.n_confs,
        n_confs_after_product_similarity_filter=args.kept,
        dist_population_size=args.target_population,
        dist_generations=args.generations,
        dist_elite_num=args.elite,
        dist_rotation_sigma=args.rotation_sigma,
        dist_translation_sigma=args.translation_sigma,
        product_similarity_coef=args.product_similarity_coef,
        forming_bond_vdw_coef=args.forming_bond_vdw_coef,
    )

    all_records = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for rxn_id in rxn_ids:
            row = df_csv[df_csv[0] == rxn_id]
            rxn_smiles = row[1].values[0]
            solvent = args.solvent
            if solvent is None:
                solvent = row[2].values[0] if df_csv.shape[1] > 2 else "water"

            print(f"\n===== Preparing reaction {rxn_id} =====")
            try:
                if not prepare_reaction(rxn_id, rxn_smiles, solvent, args, tmp_dir):
                    continue
            except Exception as e:
                print(f"[skip] rxn {rxn_id}: preparation error ({type(e).__name__}: {e})")
                continue

            for selection in selections:
                print(f"  -- EA ({selection}) ...")
                try:
                    df = run_one(rxn_id, rxn_smiles, solvent, selection, cfg, args, tmp_dir)
                    all_records.append(df)
                except Exception as e:
                    print(f"[skip] rxn {rxn_id} / {selection}: EA error ({type(e).__name__}: {e})")

    if not all_records:
        raise SystemExit("No reactions produced diversity data; nothing to plot.")

    data = pd.concat(all_records, ignore_index=True)
    csv_path = out_dir / "population_diversity.csv"
    data.to_csv(csv_path, index=False)
    print(f"\nWrote per-generation metrics -> {csv_path}")

    n_rxns = data.rxn_id.nunique()
    make_figures(data, out_dir, n_rxns, args.rmsd_cutoff, args.subsample)
    print(f"Wrote figures -> {out_dir}/diversity_fig.(pdf|png), {out_dir}/diversity_metrics.(pdf|png)")


if __name__ == "__main__":
    main()
