#!/usr/bin/env python3
"""Summarize saddle-point optimization cost for one (results folder, validator, TS method).

Reads the per-reaction ``validation_<Validator>.csv`` files that the moTSart
validator writes, restricts to IRC-validated TSs (the geometries that actually
count), and reports convergence rates plus the optimization-cycle distribution
(mean / median / MAD) and energy statistics.

This is the script that regenerates the DFT optimization cycles
numbers for any engine (``xtb`` / ``dft`` / ``mlip``) and any TS-guess source
(``racer_ts`` / ``rmsd_pp`` / ``learning``)

Examples
--------
    # baseline (RMSD-PP/racerTS guesses) optimized with the MLIP engine
    python experiments/summarize_cycles.py \
        --results-folder results_goflow/results_goflow \
        --validator MLIPValidator --ts-method racer_ts

    # TsOptNet-refined guesses, same engine
    python experiments/summarize_cycles.py \
        --results-folder results_goflow/results_goflow \
        --validator MLIPValidator --ts-method learning
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

# Reuse the exact collection logic the paper stats use.
from motsart.validator.compute_stats import (
    collect_all_results,
    get_rxn_ids_from_al_folder,
)


def cycle_stats(df: pd.DataFrame) -> dict:
    """Compute convergence + cycle + energy statistics over IRC-validated TSs."""
    n_mol = len(df)
    converged = df["converged"].mean() if n_mol else np.nan
    irc_rate = df["irc_converged"].mean() if n_mol else np.nan

    ok = df[df["irc_converged"] == True]
    if ok.empty:
        return dict(n_molecules=n_mol, converged=converged, irc_converged=irc_rate,
                    n_valid=0, cycle_mean=np.nan, cycle_median=np.nan, cycle_mad=np.nan,
                    energy_mean=np.nan, energy_median=np.nan)
    cyc = ok["cycle_cnt"].astype(float)
    return dict(
        n_molecules=n_mol,
        converged=converged,
        irc_converged=irc_rate,
        n_valid=len(ok),
        cycle_mean=cyc.mean(),
        cycle_median=cyc.median(),
        cycle_mad=(cyc - cyc.mean()).abs().mean(),
        energy_mean=ok["energy_kcal"].astype(float).mean(),
        energy_median=ok["energy_kcal"].astype(float).median(),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-folder", required=True,
                   help="Results tree containing R*/validation/<ts_method>/ ...")
    p.add_argument("--validator", default="MLIPValidator",
                   help="Validator class name used as the CSV suffix (default: MLIPValidator).")
    p.add_argument("--ts-method", default="racer_ts",
                   help="TS-guess source / path guesser (default: racer_ts).")
    p.add_argument("--rxn-ids", default=None,
                   help="Comma-separated reaction ids. Default: discover all R* folders.")
    p.add_argument("--out-csv", default=None, help="Optional path to write the summary row.")
    args = p.parse_args()

    if args.rxn_ids:
        rxn_ids = [int(x) for x in args.rxn_ids.split(",") if x.strip()]
    else:
        rxn_ids = get_rxn_ids_from_al_folder(args.results_folder)
    print(f"Found {len(rxn_ids)} reactions in {args.results_folder}")

    df = collect_all_results(rxn_ids, args.ts_method, args.validator, args.results_folder)
    if df.empty:
        print("No validation results found. Check --validator / --ts-method / --results-folder.")
        return 1

    stats = cycle_stats(df)
    stats = {"validator": args.validator, "ts_method": args.ts_method, **stats}

    print("\n=== Saddle-point optimization summary ===")
    print(f"  validator           : {stats['validator']}")
    print(f"  ts_method           : {stats['ts_method']}")
    print(f"  molecules           : {stats['n_molecules']}  (IRC-valid: {stats['n_valid']})")
    print(f"  converged rate      : {stats['converged']:.3f}")
    print(f"  IRC-converged rate  : {stats['irc_converged']:.3f}")
    print(f"  opt cycles  mean    : {stats['cycle_mean']:.3f}")
    print(f"  opt cycles  median  : {stats['cycle_median']:.1f}")
    print(f"  opt cycles  MAD     : {stats['cycle_mad']:.3f}")
    print(f"  energy kcal median  : {stats['energy_median']:.3f}")

    if args.out_csv:
        pd.DataFrame([stats]).to_csv(args.out_csv, index=False)
        print(f"\nWrote summary -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
