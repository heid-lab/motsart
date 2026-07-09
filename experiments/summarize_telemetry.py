#!/usr/bin/env python3
"""Aggregate per-reaction telemetry (timings + failure tallies)

Reads every ``R*/telemetry/metrics.jsonl`` under a results tree and reports timings and failure rates.

Records are written by ``motsart.telemetry`` (see that module for the schema).

Example
-------
    python experiments/summarize_telemetry.py \
        --results-folder results_goflow/results_goflow \
        --out-prefix results_goflow/telemetry
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRACKED_COMPONENTS = {
    "complex_finder_total", "afir_total",
    "path_search_total",
    "racer_ts_total",
    "sp_opt", "irc",
}


def load_records(results_folder: str, rxn_ids=None) -> pd.DataFrame:
    root = Path(results_folder)
    files = sorted(root.glob("**/R*/telemetry/metrics.jsonl"))
    rows = []
    for fp in files:
        for line in fp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(rec)
    df = pd.DataFrame(rows)
    if not df.empty and rxn_ids is not None:
        df = df[df["rxn_id"].isin(rxn_ids)]
    return df


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def summarize_timing(df: pd.DataFrame, out_prefix=None) -> None:
    t = df[df["kind"] == "timing"].copy()
    if t.empty:
        print("\n[timing] no timing records found.")
        return
    t["seconds"] = pd.to_numeric(t["seconds"], errors="coerce")

    by_comp = (t.groupby(["stage", "component"])["seconds"]
                 .agg(n="count", total_s="sum", mean_s="mean", median_s="median", max_s="max")
                 .reset_index().sort_values(["stage", "total_s"], ascending=[True, False]))
    print("\n=== Timing by component (all timers; leaves nested in *_total) ===")
    print(_fmt(by_comp))

    tracked_comp = t[t["component"].isin(TRACKED_COMPONENTS)]
    if not tracked_comp.empty:
        stage_tot = (tracked_comp.groupby("stage")["seconds"].sum()
                     .reset_index().rename(columns={"seconds": "total_s"}))
        grand = stage_tot["total_s"].sum()
        stage_tot["share_%"] = 100.0 * stage_tot["total_s"] / grand if grand else np.nan
        print("\n=== Wall-clock by stage (non-overlapping; end-to-end) ===")
        print(_fmt(stage_tot.sort_values("total_s", ascending=False)))

        per_rxn = tracked_comp.groupby("rxn_id")["seconds"].sum()
        print("\n=== End-to-end wall-clock per reaction (seconds) ===")
        print(f"  reactions : {per_rxn.shape[0]}")
        print(f"  total     : {per_rxn.sum():.1f}")
        print(f"  mean      : {per_rxn.mean():.1f}")
        print(f"  median    : {per_rxn.median():.1f}")
        print(f"  min / max : {per_rxn.min():.1f} / {per_rxn.max():.1f}")

        ev = df[(df["kind"] == "event") & (df["component"] == "irc")]
        n_valid = int((pd.to_numeric(ev.get("total", 0)) - pd.to_numeric(ev.get("failed", 0))).sum()) if not ev.empty else 0
        val_s = t[t["component"].isin(["sp_opt", "irc"])]["seconds"].sum()
        if n_valid:
            print(f"\n  validator seconds / IRC-valid TS : {val_s / n_valid:.1f}  "
                  f"({val_s:.0f}s over {n_valid} validated TSs)")

    if out_prefix:
        by_comp.to_csv(f"{out_prefix}_timing_by_component.csv", index=False)
        print(f"\nWrote {out_prefix}_timing_by_component.csv")


def summarize_failures(df: pd.DataFrame, out_prefix=None) -> None:
    e = df[df["kind"] == "event"].copy()
    if e.empty:
        print("\n[failures] no event records found.")
        return
    e["total"] = pd.to_numeric(e["total"], errors="coerce").fillna(0)
    e["failed"] = pd.to_numeric(e["failed"], errors="coerce").fillna(0)

    by = (e.groupby(["stage", "component"]).agg(attempts=("total", "sum"),failed=("failed", "sum"), n_reactions=("rxn_id", "nunique")).reset_index())
    by["failure_rate_%"] = 100.0 * by["failed"] / by["attempts"].replace(0, np.nan)
    by["pass_rate_%"] = 100.0 - by["failure_rate_%"]
    by = by.sort_values(["stage", "component"])
    print("\n=== Failure / pass rates by stage and component ===")
    print(_fmt(by))

    if out_prefix:
        by.to_csv(f"{out_prefix}_failures.csv", index=False)
        # Also a per-reaction breakdown for seed-variance / robustness reporting.
        per_rxn = (e.groupby(["rxn_id", "stage", "component"]).agg(attempts=("total", "sum"), failed=("failed", "sum")).reset_index())
        per_rxn.to_csv(f"{out_prefix}_failures_per_reaction.csv", index=False)
        print(f"Wrote {out_prefix}_failures.csv and {out_prefix}_failures_per_reaction.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-folder", required=True, help="Results tree (searched recursively for R*/telemetry/metrics.jsonl).")
    p.add_argument("--rxn-ids", default=None, help="Comma-separated reaction ids to include (default: all).")
    p.add_argument("--out-prefix", default=None, help="If set, write summary CSVs with this path prefix.")
    args = p.parse_args()

    rxn_ids = [int(x) for x in args.rxn_ids.split(",")] if args.rxn_ids else None
    df = load_records(args.results_folder, rxn_ids)
    if df.empty:
        print(f"No telemetry records found under {args.results_folder}. "
              f"Run the pipeline with MOTSART_TELEMETRY != 0 first.")
        return 1
    n_rxn = df["rxn_id"].nunique()
    print(f"Loaded {len(df)} telemetry records across {n_rxn} reactions.")

    summarize_timing(df, args.out_prefix)
    summarize_failures(df, args.out_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
