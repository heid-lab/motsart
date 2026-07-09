#!/usr/bin/env python3
"""Compare E2 H-Cb-Ca-LG dihedral distributions: all attempted vs. solved TSs.

For every E2 candidate that reached the validator, measures the H-Cb-Ca-LG
dihedral on the initial ML-FSM TS guess (pre-optimization). For every
IRC-validated ("solved") TS, measures the same dihedral on the optimized
saddle-point geometry. Plots both distributions side by side (15-degree bins)
and reports, for reactions where both a syn-periplanar and an anti-periplanar
TS were validated, a head-to-head comparison of their energies.

Example
-------
    python experiments/e2_periplanar_analysis.py \
        --results-folder results_sn2e2_mlfsm_v2 --ts-method ml_fsm --validator MLIPValidator
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read as ase_read

sys.path.insert(0, str(Path(__file__).parent))
from check_e2_anti_periplanar import compute_dihedral, find_e2_core_atoms, ts_geometry_path

from motsart.complex_finder.utils import get_rxn_data
from motsart.validator.compute_stats import collect_all_results, get_rxn_ids_from_al_folder


def guess_geometry_path(results_folder: str, rxn_id: int, ts_method: str, ts_file: str) -> Path:
    return Path(results_folder) / f"R{rxn_id}" / "ts" / ts_method / "ts_to_validate" / ts_file


def compute_dihedrals_for_rows(rows_df: pd.DataFrame, geometry_fn, rxn_data_cache: dict,
                                smiles_map: pd.Series) -> pd.DataFrame:
    """Compute the E2 core dihedral for every (rxn_id, ts_file) row using geometry_fn(row) -> Path."""
    out = []
    for _, row in rows_df.iterrows():
        rxn_id, ts_file = int(row.rxn_id), row.ts_file
        if rxn_id not in rxn_data_cache:
            rxn_data_cache[rxn_id] = get_rxn_data(rxn_id, smiles_map[rxn_id], r_or_p="r")
        rd = rxn_data_cache[rxn_id]

        core = find_e2_core_atoms(rd)
        if core is None:
            continue
        xyz_path = geometry_fn(rxn_id, ts_file)
        if not xyz_path.exists():
            continue

        atoms = ase_read(str(xyz_path))
        coords = atoms.get_positions()
        idx = {name: rd.mn_order[mn] for name, mn in
               [("h", core["h_mn"]), ("cb", core["cb_mn"]), ("ca", core["ca_mn"]), ("lg", core["lg_mn"])]}
        dih = compute_dihedral(coords[idx["h"]], coords[idx["cb"]], coords[idx["ca"]], coords[idx["lg"]])

        rec = row.to_dict()
        rec["dihedral_deg"] = dih
        rec["abs_dihedral_deg"] = abs(dih)
        out.append(rec)
    return pd.DataFrame(out)


def compute_reactant_complex_dihedrals(rxn_ids: list, results_folder: str, rxn_data_cache: dict,
                                        smiles_map: pd.Series) -> pd.DataFrame:
    """Compute the E2 core dihedral on every final reactant-complex geometry (pre-AFIR, pre-ML-FSM).

    Unlike the TS-guess dihedral, this is available for any reaction where complex_finder
    produced at least one reactant complex -- regardless of whether AFIR/ML-FSM later
    succeeded -- so restricting on it doesn't silently drop every path-search failure.
    """
    out = []
    for rxn_id in rxn_ids:
        if rxn_id not in rxn_data_cache:
            rxn_data_cache[rxn_id] = get_rxn_data(rxn_id, smiles_map[rxn_id], r_or_p="r")
        rd = rxn_data_cache[rxn_id]

        core = find_e2_core_atoms(rd)
        if core is None:
            continue
        rc_dir = Path(results_folder) / f"R{rxn_id}" / "r" / "final_complexes"
        if not rc_dir.exists():
            continue

        idx = {name: rd.mn_order[mn] for name, mn in
               [("h", core["h_mn"]), ("cb", core["cb_mn"]), ("ca", core["ca_mn"]), ("lg", core["lg_mn"])]}
        for rc_path in sorted(rc_dir.glob("*.xyz")):
            atoms = ase_read(str(rc_path))
            coords = atoms.get_positions()
            dih = compute_dihedral(coords[idx["h"]], coords[idx["cb"]], coords[idx["ca"]], coords[idx["lg"]])
            out.append({"rxn_id": rxn_id, "rc_file": rc_path.name,
                        "dihedral_deg": dih, "abs_dihedral_deg": abs(dih)})
    return pd.DataFrame(out)


def print_histogram(df: pd.DataFrame, title: str, bin_width: float) -> pd.Series:
    bins = list(np.arange(0, 180 + bin_width, bin_width))
    labels = [f"{int(a)}-{int(b)}" for a, b in zip(bins[:-1], bins[1:])]
    counts = pd.cut(df.abs_dihedral_deg, bins=bins, labels=labels, right=False).value_counts().sort_index()
    print(f"\n=== {title} (n={len(df)}) ===")
    for label, n in counts.items():
        print(f"  {label:>8s}: {n:3d} {'#' * int(n)}")
    return counts


def classify_reaction(sub: pd.DataFrame) -> str:
    """First pipeline stage a reaction fails at, matching the paper's table categories."""
    if len(sub) == 0:
        return "no_guess"
    if (sub.irc_converged == True).any():
        return "solved"
    if (sub.converged == True).any():
        return "saddle_irc_fail"
    return "none_conv"


def compute_e2_lg_base_features(mech_rxn_ids: list, mech_results: pd.DataFrame, smiles_map: pd.Series,
                                 rxn_data_cache: dict) -> pd.DataFrame:
    """Leaving-group / base element identity for every E2 reaction, plus its reaction-level outcome."""
    rows = []
    for rxn_id in mech_rxn_ids:
        if rxn_id not in rxn_data_cache:
            rxn_data_cache[rxn_id] = get_rxn_data(rxn_id, smiles_map[rxn_id], r_or_p="r")
        rd = rxn_data_cache[rxn_id]
        core = find_e2_core_atoms(rd)
        if core is None:
            continue
        lg_elem = rd.r_mol.GetAtomWithIdx(rd.r_mn_to_idx_dict[core["lg_mn"]]).GetSymbol()
        base_mn = next(x for b in rd.formed_bonds_mn_Bf if core["h_mn"] in b for x in b if x != core["h_mn"])
        base_elem = rd.r_mol.GetAtomWithIdx(rd.r_mn_to_idx_dict[base_mn]).GetSymbol()
        outcome = classify_reaction(mech_results[mech_results.rxn_id == rxn_id])
        rows.append({"rxn_id": rxn_id, "outcome": outcome, "lg_elem": lg_elem, "base_elem": base_elem})
    return pd.DataFrame(rows)


def print_lg_base_analysis(lg_base: pd.DataFrame) -> None:
    """Leaving-group / base element breakdown by outcome, and the fluorine-involvement success gap."""
    print(f"\n=== E2 leaving-group / base element analysis (n={len(lg_base)}) ===")
    print("\n  LG element by outcome:")
    print(pd.crosstab(lg_base.outcome, lg_base.lg_elem).to_string())
    print("\n  Base element by outcome:")
    print(pd.crosstab(lg_base.outcome, lg_base.base_elem).to_string())

    print("\n  Success rate by base element:")
    for elem in sorted(lg_base.base_elem.unique()):
        sub = lg_base[lg_base.base_elem == elem]
        n_irc_fail = (sub.outcome == "saddle_irc_fail").sum()
        print(f"    base={elem:3s} n={len(sub):3d}  IRC-fail={n_irc_fail:3d} ({100*n_irc_fail/len(sub):.1f}%)  "
              f"solved={100*(sub.outcome=='solved').mean():.1f}%")

    f_either = lg_base[(lg_base.lg_elem == "F") | (lg_base.base_elem == "F")]
    f_none = lg_base[(lg_base.lg_elem != "F") & (lg_base.base_elem != "F")]
    print(f"\n  F as LG or base : n={len(f_either):3d}  solved={(f_either.outcome=='solved').sum():3d} "
          f"({100*(f_either.outcome=='solved').mean():.1f}%)")
    print(f"  F in neither role: n={len(f_none):3d}  solved={(f_none.outcome=='solved').sum():3d} "
          f"({100*(f_none.outcome=='solved').mean():.1f}%)")


def compute_sn2_lg_nu_features(mech_rxn_ids: list, mech_results: pd.DataFrame, smiles_map: pd.Series,
                                rxn_data_cache: dict) -> pd.DataFrame:
    """Leaving-group / nucleophile element identity for every SN2 reaction, plus its outcome.

    SN2 has a single broken bond (Ca-LG) and a single formed bond (Ca-Nu); Ca is the atom
    common to both.
    """
    rows = []
    for rxn_id in mech_rxn_ids:
        if rxn_id not in rxn_data_cache:
            rxn_data_cache[rxn_id] = get_rxn_data(rxn_id, smiles_map[rxn_id], r_or_p="r")
        rd = rxn_data_cache[rxn_id]
        broken = list(rd.broken_bonds_mn_Bf)
        formed = list(rd.formed_bonds_mn_Bf)
        if len(broken) != 1 or len(formed) != 1:
            continue
        ca_mn = set(broken[0]) & set(formed[0])
        if len(ca_mn) != 1:
            continue
        ca_mn = ca_mn.pop()
        lg_mn = [m for m in broken[0] if m != ca_mn][0]
        nu_mn = [m for m in formed[0] if m != ca_mn][0]
        lg_elem = rd.r_mol.GetAtomWithIdx(rd.r_mn_to_idx_dict[lg_mn]).GetSymbol()
        nu_elem = rd.r_mol.GetAtomWithIdx(rd.r_mn_to_idx_dict[nu_mn]).GetSymbol()
        outcome = classify_reaction(mech_results[mech_results.rxn_id == rxn_id])
        rows.append({"rxn_id": rxn_id, "outcome": outcome, "lg_elem": lg_elem, "nu_elem": nu_elem})
    return pd.DataFrame(rows)


def funnel_row(rxn_ids_subset, mech_results: pd.DataFrame) -> dict:
    cats = [classify_reaction(mech_results[mech_results.rxn_id == r]) for r in rxn_ids_subset]
    n = len(rxn_ids_subset)
    counts = {c: cats.count(c) for c in ["no_guess", "none_conv", "saddle_irc_fail", "solved"]}
    success = 100 * counts["solved"] / n if n else float("nan")
    return {"n": n, **counts, "success_%": success}


def print_mechanism_outcome_table(mech_rxn_ids: list, mech_results: pd.DataFrame,
                                   variant_rows: dict, mechanism_label: str) -> None:
    """Reaction-level pipeline outcome, matching the paper's table columns.

    Baseline row uses all reactions of this mechanism (should reproduce the paper's
    published Path search/SP opt./IRC/Solved/Success numbers exactly). Each entry in
    variant_rows is (row label -> set of rxn_ids to restrict N to); the numerator always
    uses the standard reaction-level "solved" definition (ANY candidate validated).
    """
    print(f"\n=== Pipeline outcome table (reaction-level; reproduces the paper's Table) ===")
    hdr = f"{'Rxn':28s} {'N':>4s} {'PathSearch':>10s} {'SPopt':>6s} {'IRC':>4s} {'Solved':>7s} {'Success':>8s}"
    print(hdr)

    base = funnel_row(mech_rxn_ids, mech_results)
    print(f"{mechanism_label.upper():28s} {base['n']:>4d} {base['no_guess']:>10d} {base['none_conv']:>6d} "
          f"{base['saddle_irc_fail']:>4d} {base['solved']:>7d} {base['success_%']:>7.1f}%")

    for row_label, rxn_subset in variant_rows.items():
        ids = [r for r in mech_rxn_ids if r in rxn_subset]
        row = funnel_row(ids, mech_results)
        print(f"{row_label:28s} {row['n']:>4d} {row['no_guess']:>10d} {row['none_conv']:>6d} "
              f"{row['saddle_irc_fail']:>4d} {row['solved']:>7d} {row['success_%']:>7.1f}%")

    print(f"\n  Each restricted row keeps N to a subset of reactions (anti-periplanar candidate/complex "
          f"present, or F- excluded from LG/base); a reaction still counts as solved if ANY of its "
          f"candidates validated.")


def plot_comparison(attempted_counts: pd.Series, solved_counts: pd.Series, out_path: Path, bin_width: float) -> None:
    labels = list(attempted_counts.index)
    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, attempted_counts.values, width, label="All attempted (TS guess)", color="#7f8fa6")
    ax.bar(x + width / 2, solved_counts.reindex(attempted_counts.index, fill_value=0).values, width,
           label="Solved (Validated TS)", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("|H-Cb-Ca-LG dihedral| (deg)")
    ax.set_ylabel("count")
    ax.legend()
    ax.axvspan(-0.5, 45 / bin_width - 0.5, color="tab:blue", alpha=0.05)
    ax.axvspan(len(labels) - 45 / bin_width - 0.5, len(labels) - 0.5, color="tab:orange", alpha=0.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"\nWrote plot -> {out_path}")
    print(f"Wrote plot -> {pdf_path}")


def energy_comparison(solved_dih: pd.DataFrame, syn_threshold: float, anti_threshold: float) -> None:
    df = solved_dih.copy()
    df["type"] = np.where(df.abs_dihedral_deg >= anti_threshold, "anti",
                           np.where(df.abs_dihedral_deg <= syn_threshold, "syn", "other"))
    typed = df[df.type != "other"]

    print(f"\n=== Energy comparison: anti (>= {anti_threshold:.0f} deg) vs syn (<= {syn_threshold:.0f} deg) ===")
    print(f"  mean  energy_kcal : anti={typed[typed.type=='anti'].energy_kcal.mean():.2f}   "
          f"syn={typed[typed.type=='syn'].energy_kcal.mean():.2f}")
    print(f"  median energy_kcal: anti={typed[typed.type=='anti'].energy_kcal.median():.2f}   "
          f"syn={typed[typed.type=='syn'].energy_kcal.median():.2f}")

    both = typed.groupby("rxn_id").type.nunique()
    rxns_both = both[both == 2].index
    print(f"\n  reactions with BOTH anti and syn validated TSs: {len(rxns_both)}")
    n_anti_lower = 0
    diffs = []
    for r in rxns_both:
        g = typed[typed.rxn_id == r]
        e_anti = g[g.type == "anti"].energy_kcal.min()
        e_syn = g[g.type == "syn"].energy_kcal.min()
        winner = "anti" if e_anti < e_syn else "syn"
        n_anti_lower += winner == "anti"
        diffs.append(e_syn - e_anti)
        print(f"    rxn {r:3d}: anti={e_anti:14.3f}  syn={e_syn:14.3f}  "
              f"diff(syn-anti)={e_syn - e_anti:+8.2f} kcal  -> lower: {winner}")
    if len(rxns_both):
        print(f"\n  anti lower in {n_anti_lower}/{len(rxns_both)} head-to-head comparisons")
        diffs = np.array(diffs)
        print(f"  mean  energy diff (syn-anti): {diffs.mean():+.2f} kcal  (std={diffs.std():.2f})")
        print(f"  median energy diff (syn-anti): {np.median(diffs):+.2f} kcal")

    per_rxn_mode = typed.groupby("rxn_id").type.agg(
        lambda s: "all_anti" if (s == "anti").all() else ("all_syn" if (s == "syn").all() else "mixed"))
    print(f"\n  solved-reaction breakdown: {per_rxn_mode.value_counts().to_dict()}")
    only_syn = sorted(per_rxn_mode[per_rxn_mode == "all_syn"].index.tolist())
    print(f"  solved ONLY via syn (never found the lower-energy anti path): {len(only_syn)} -> {only_syn}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-folder", required=True)
    p.add_argument("--ts-method", default="ml_fsm")
    p.add_argument("--validator", default="MLIPValidator")
    p.add_argument("--rxn-csv", default="data/reactions_sn2_e2.csv")
    p.add_argument("--metadata-csv", default="data/reactions_sn2_e2_metadata.csv")
    p.add_argument("--mechanism", default="e2")
    p.add_argument("--bin-width", type=float, default=15.0)
    p.add_argument("--syn-threshold", type=float, default=45.0)
    p.add_argument("--anti-threshold", type=float, default=150.0)
    p.add_argument("--out-prefix", default=None, help="Prefix for the plot/CSV outputs (default: <results-folder>/e2_periplanar)")
    args = p.parse_args()

    out_prefix = args.out_prefix or str(Path(args.results_folder) / "e2_periplanar")

    smiles_map = pd.read_csv(args.rxn_csv, header=None).set_index(0)[1]
    meta = pd.read_csv(args.metadata_csv).set_index("id")["reaction"]

    rxn_ids = get_rxn_ids_from_al_folder(args.results_folder)
    mech_rxn_ids = [r for r in rxn_ids if meta.get(r) == args.mechanism]
    all_results = collect_all_results(rxn_ids, args.ts_method, args.validator, args.results_folder)
    all_results["mech"] = all_results.rxn_id.map(meta)
    mech_results = all_results[all_results.mech == args.mechanism]

    attempted = mech_results[["rxn_id", "ts_file", "converged", "irc_converged"]]
    solved = mech_results[mech_results.irc_converged == True][["rxn_id", "ts_file", "converged", "irc_converged", "energy_kcal"]]

    rxn_data_cache: dict = {}
    variant_rows = {}
    lg_base = None
    sn2_lg_nu = None

    if args.mechanism == "e2":
        attempted_dih = compute_dihedrals_for_rows(
            attempted, lambda r, f: guess_geometry_path(args.results_folder, r, args.ts_method, f),
            rxn_data_cache, smiles_map)
        solved_dih = compute_dihedrals_for_rows(
            solved, lambda r, f: ts_geometry_path(args.results_folder, r, args.ts_method, args.validator, f),
            rxn_data_cache, smiles_map)

        print(f"Attempted {args.mechanism.upper()} candidates evaluated: {len(attempted_dih)} / {len(attempted)}")
        print(f"Solved {args.mechanism.upper()} TSs evaluated         : {len(solved_dih)} / {len(solved)}")

        attempted_counts = print_histogram(attempted_dih, "All attempted (initial ML-FSM guess)", args.bin_width)
        solved_counts = print_histogram(solved_dih, "Solved (optimized, IRC-validated TS)", args.bin_width)

        rc_dih = compute_reactant_complex_dihedrals(mech_rxn_ids, args.results_folder, rxn_data_cache, smiles_map)
        guess_anti_rxns = set(attempted_dih[attempted_dih.abs_dihedral_deg >= args.anti_threshold].rxn_id.unique())
        rc_anti_rxns = set(rc_dih[rc_dih.abs_dihedral_deg >= args.anti_threshold].rxn_id.unique())
        variant_rows[f"{args.mechanism.upper()}-prior (TS guess)"] = guess_anti_rxns
        variant_rows[f"{args.mechanism.upper()}-prior (reactant complex)"] = rc_anti_rxns

        lg_base = compute_e2_lg_base_features(mech_rxn_ids, mech_results, smiles_map, rxn_data_cache)
        f_none_rxns = set(lg_base[(lg_base.lg_elem != "F") & (lg_base.base_elem != "F")].rxn_id.unique())
        variant_rows[f"{args.mechanism.upper()}-No-F⁻"] = f_none_rxns

    elif args.mechanism == "sn2":
        sn2_lg_nu = compute_sn2_lg_nu_features(mech_rxn_ids, mech_results, smiles_map, rxn_data_cache)
        no_f_lg_rxns = set(sn2_lg_nu[sn2_lg_nu.lg_elem != "F"].rxn_id.unique())
        variant_rows[f"{args.mechanism.upper()}-No-F⁻-LG"] = no_f_lg_rxns

    print_mechanism_outcome_table(mech_rxn_ids, mech_results, variant_rows, args.mechanism)

    if args.mechanism == "e2":
        attempted_dih.to_csv(f"{out_prefix}_attempted.csv", index=False)
        solved_dih.to_csv(f"{out_prefix}_solved.csv", index=False)
        rc_dih.to_csv(f"{out_prefix}_reactant_complex.csv", index=False)
        print(f"\nWrote {out_prefix}_attempted.csv, {out_prefix}_solved.csv, and {out_prefix}_reactant_complex.csv")

        plot_comparison(attempted_counts, solved_counts, Path(f"{out_prefix}_histogram.png"), args.bin_width)

        energy_comparison(solved_dih, args.syn_threshold, args.anti_threshold)

    if lg_base is not None:
        print_lg_base_analysis(lg_base)
        lg_base.to_csv(f"{out_prefix}_lg_base.csv", index=False)
        print(f"\nWrote {out_prefix}_lg_base.csv")

    if sn2_lg_nu is not None:
        sn2_lg_nu.to_csv(f"{out_prefix}_lg_nu.csv", index=False)
        print(f"\nWrote {out_prefix}_lg_nu.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
