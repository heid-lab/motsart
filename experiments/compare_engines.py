#!/usr/bin/env python3
"""Compare saddle-point optimization cost across engines and/or TS-guess sources.

Each series is a (results-folder, validator, ts-method) triple. The script
computes the same cycle/convergence/energy statistics for every series over the
shared set of IRC-validated molecules (matched by rxn_id + ts_file so the
comparison is paired), then reports the reduction of every series relative to
the first (baseline).

Series syntax (repeatable):  LABEL:RESULTS_FOLDER:VALIDATOR:TS_METHOD

Examples
--------
    # TsOptNet vs baseline, both optimized with the MLIP engine
    python experiments/compare_engines.py \
        --series baseline:results_goflow/results_goflow:MLIPValidator:racer_ts \
        --series tsoptnet:results_goflow/results_goflow:MLIPValidator:learning \
        --out-csv results_goflow/compare_mlip.csv

    # Same guesses, DFT engine vs MLIP engine (engine swap)
    python experiments/compare_engines.py \
        --series dft:results_dft/results_dft:DFTValidator:racer_ts \
        --series mlip:results_mlip/results_mlip:MLIPValidator:racer_ts
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from motsart.validator.compute_stats import (
    collect_all_results,
    get_rxn_ids_from_al_folder,
)


def parse_series(spec: str):
    parts = spec.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--series must be LABEL:RESULTS_FOLDER:VALIDATOR:TS_METHOD, got: {spec!r}")
    label, folder, validator, ts_method = parts
    return dict(label=label, folder=folder, validator=validator, ts_method=ts_method)


def load_series(s: dict, rxn_ids=None) -> pd.DataFrame:
    ids = rxn_ids or get_rxn_ids_from_al_folder(s["folder"])
    df = collect_all_results(ids, s["ts_method"], s["validator"], s["folder"])
    if df.empty:
        return df
    df = df.copy()
    df["__key"] = df["rxn_id"].astype(str) + "/" + df["ts_file"].astype(str)
    return df


_STAT_KEYS = ("n_molecules", "n_valid", "converged", "irc_converged",
              "cycle_mean", "cycle_median", "cycle_mad", "energy_median")


def stats_over(df: pd.DataFrame) -> dict:
    needed = {"converged", "irc_converged", "cycle_cnt", "energy_kcal"}
    if df.empty or not needed <= set(df.columns):
        return {k: (0 if k in ("n_molecules", "n_valid") else np.nan) for k in _STAT_KEYS}
    ok = df[df["irc_converged"] == True]
    cyc = ok["cycle_cnt"].astype(float)
    return dict(
        n_molecules=len(df),
        n_valid=len(ok),
        converged=df["converged"].mean(),
        irc_converged=df["irc_converged"].mean(),
        cycle_mean=cyc.mean() if len(ok) else np.nan,
        cycle_median=cyc.median() if len(ok) else np.nan,
        cycle_mad=(cyc - cyc.mean()).abs().mean() if len(ok) else np.nan,
        energy_median=ok["energy_kcal"].astype(float).median() if len(ok) else np.nan,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", action="append", type=parse_series, required=True,
                   help="LABEL:RESULTS_FOLDER:VALIDATOR:TS_METHOD (repeatable).")
    p.add_argument("--paired", action="store_true",
                   help="Restrict every series to molecules present (IRC-valid) in ALL series.")
    p.add_argument("--out-csv", default=None, help="Write the comparison table here.")
    args = p.parse_args()

    loaded = [(s, load_series(s)) for s in args.series]
    for s, df in loaded:
        if df.empty:
            print(f"[warn] series '{s['label']}' has no results "
                  f"({s['folder']} / {s['validator']} / {s['ts_method']})")

    if args.paired:
        # Intersection of IRC-valid molecule keys across all non-empty series.
        valid_keys = None
        for _s, df in loaded:
            if df.empty:
                continue
            keys = set(df[df["irc_converged"] == True]["__key"])
            valid_keys = keys if valid_keys is None else (valid_keys & keys)
        valid_keys = valid_keys or set()
        loaded = [(s, df[df["__key"].isin(valid_keys)] if not df.empty else df) for s, df in loaded]
        print(f"Paired comparison over {len(valid_keys)} shared IRC-valid molecules.")

    rows = []
    for s, df in loaded:
        st = stats_over(df)  # always returns the full key set (NaNs if empty)
        rows.append({"label": s["label"], "validator": s["validator"],
                     "ts_method": s["ts_method"], **st})
    table = pd.DataFrame(rows)

    # Reduction relative to the first series (baseline).
    base = table.iloc[0]
    def red(col):
        b = base[col]
        return [np.nan if (pd.isna(b) or b == 0 or pd.isna(v)) else 100.0 * (b - v) / b
                for v in table[col]]
    table["cycle_mean_reduction_%"] = red("cycle_mean")
    table["cycle_median_reduction_%"] = red("cycle_median")

    pd.set_option("display.width", 160, "display.max_columns", 30)
    print("\n=== Engine / guess comparison (reduction vs first series) ===")
    cols = ["label", "validator", "ts_method", "n_valid", "converged", "irc_converged",
            "cycle_mean", "cycle_median", "cycle_mad",
            "cycle_mean_reduction_%", "cycle_median_reduction_%"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.out_csv:
        table.to_csv(args.out_csv, index=False)
        print(f"\nWrote comparison -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
