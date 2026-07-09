#!/usr/bin/env python3
"""Check the H-C(beta)-C(alpha)-LG dihedral of validated E2 TS guesses.

Example
-------
    python experiments/check_e2_anti_periplanar.py \
        --results-folder results_sn2e2_mlfsm --ts-method ml_fsm --validator MLIPValidator
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read as ase_read

from motsart.complex_finder.utils import get_rxn_data
from motsart.validator.compute_stats import collect_all_results, get_rxn_ids_from_al_folder

ANTI_THRESHOLD_DEG = 150.0


def compute_dihedral(p0, p1, p2, p3) -> float:
    """Dihedral angle (degrees, in [-180, 180]) for four points."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def find_e2_core_atoms(rxn_data) -> dict | None:
    """Identify (Ca, Cb, H_migrating, LG) atom-map numbers for an E2 reaction.

    Uses the bonds broken/formed between reactant and product:
    - one broken bond not involving H -> (Ca, LG)
    - one broken bond involving H -> (Cb, H)
    - Ca and Cb must be adjacent (bonded) in the reactant.
    Returns None if the reaction doesn't match this simple E2 pattern.
    """
    broken = list(rxn_data.broken_bonds_mn_Bf)
    if len(broken) != 2:
        return None

    def symbol(mn):
        idx = rxn_data.r_mn_to_idx_dict[mn]
        return rxn_data.r_mol.GetAtomWithIdx(idx).GetSymbol()

    h_bond = [b for b in broken if any(symbol(mn) == "H" for mn in b)]
    other_bond = [b for b in broken if b not in h_bond]
    if len(h_bond) != 1 or len(other_bond) != 1:
        return None

    h_mn = next(mn for mn in h_bond[0] if symbol(mn) == "H")
    cb_mn = next(mn for mn in h_bond[0] if mn != h_mn)

    ca_lg = other_bond[0]
    non_carbon = [mn for mn in ca_lg if symbol(mn) != "C"]
    if len(non_carbon) != 1:
        return None
    lg_mn = non_carbon[0]
    ca_mn = next(mn for mn in ca_lg if mn != lg_mn)

    ca_idx = rxn_data.r_mn_to_idx_dict[ca_mn]
    cb_idx = rxn_data.r_mn_to_idx_dict[cb_mn]
    if rxn_data.r_mol.GetBondBetweenAtoms(ca_idx, cb_idx) is None:
        return None  # Ca/Cb must be adjacent for a concerted E2 core

    return {"ca_mn": ca_mn, "cb_mn": cb_mn, "h_mn": h_mn, "lg_mn": lg_mn}


def ts_geometry_path(results_folder: str, rxn_id: int, ts_method: str, validator: str, ts_file: str) -> Path:
    stem = Path(ts_file).stem
    return (Path(results_folder) / f"R{rxn_id}" / "validation" / ts_method /
            "ts_sp_opt" / f"orca_{validator}" / f"{stem}.ts_freq.xyz")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-folder", required=True)
    p.add_argument("--ts-method", default="ml_fsm")
    p.add_argument("--validator", default="MLIPValidator")
    p.add_argument("--rxn-csv", default="data/reactions_sn2_e2.csv")
    p.add_argument("--metadata-csv", default="data/reactions_sn2_e2_metadata.csv")
    p.add_argument("--anti-threshold", type=float, default=ANTI_THRESHOLD_DEG,
                    help="abs(dihedral) >= this (degrees) counts as anti-periplanar")
    args = p.parse_args()

    smiles_map = pd.read_csv(args.rxn_csv, header=None).set_index(0)[1]
    mechanism = pd.read_csv(args.metadata_csv).set_index("id")["reaction"]

    rxn_ids = get_rxn_ids_from_al_folder(args.results_folder)
    df = collect_all_results(rxn_ids, args.ts_method, args.validator, args.results_folder)
    e2_ids = [i for i in rxn_ids if mechanism.get(i) == "e2"]
    solved = df[(df.rxn_id.isin(e2_ids)) & (df.irc_converged == True)]

    print(f"Checking {len(solved)} IRC-validated E2 TS geometries "
          f"(from {len(e2_ids)} E2 reactions) for anti-periplanar H-Cb-Ca-LG...\n")

    rows = []
    rxn_data_cache: dict[int, object] = {}
    for _, row in solved.iterrows():
        rxn_id, ts_file = int(row.rxn_id), row.ts_file
        if rxn_id not in rxn_data_cache:
            rxn_data_cache[rxn_id] = get_rxn_data(rxn_id, smiles_map[rxn_id], r_or_p="r")
        rxn_data = rxn_data_cache[rxn_id]

        core = find_e2_core_atoms(rxn_data)
        if core is None:
            print(f"  R{rxn_id}/{ts_file}: could not identify E2 core atoms, skipping")
            continue

        xyz_path = ts_geometry_path(args.results_folder, rxn_id, args.ts_method, args.validator, ts_file)
        if not xyz_path.exists():
            print(f"  R{rxn_id}/{ts_file}: TS geometry not found at {xyz_path}, skipping")
            continue

        atoms = ase_read(str(xyz_path))
        coords = atoms.get_positions()
        idx = {name: rxn_data.mn_order[mn] for name, mn in
               [("h", core["h_mn"]), ("cb", core["cb_mn"]), ("ca", core["ca_mn"]), ("lg", core["lg_mn"])]}

        dihedral = compute_dihedral(coords[idx["h"]], coords[idx["cb"]], coords[idx["ca"]], coords[idx["lg"]])
        rows.append({"rxn_id": rxn_id, "ts_file": ts_file, "dihedral_deg": dihedral,
                      "abs_dihedral_deg": abs(dihedral)})

    if not rows:
        print("No TS geometries could be evaluated.")
        return 1

    result = pd.DataFrame(rows)
    is_anti = result.abs_dihedral_deg >= args.anti_threshold
    n_anti = int(is_anti.sum())
    n_total = len(result)

    print(result.sort_values("abs_dihedral_deg").to_string(index=False, float_format=lambda x: f"{x:6.1f}"))
    print(f"\n=== Summary ===")
    print(f"  evaluated TSs           : {n_total}")
    print(f"  anti-periplanar (>= {args.anti_threshold:.0f} deg): {n_anti} ({100 * n_anti / n_total:.1f}%)")
    print(f"  mean |dihedral|         : {result.abs_dihedral_deg.mean():.1f} deg")
    print(f"  median |dihedral|       : {result.abs_dihedral_deg.median():.1f} deg")

    bins = [0, 30, 60, 90, 120, 150, 180.01]
    labels = ["0-30", "30-60", "60-90", "90-120", "120-150", "150-180"]
    hist = pd.cut(result.abs_dihedral_deg, bins=bins, labels=labels, right=False).value_counts().sort_index()
    print("\n  |dihedral| histogram (deg):")
    for label, count in hist.items():
        print(f"    {label:8s}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
