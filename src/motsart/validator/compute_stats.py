"""Aggregate and compare validation statistics across TS methods.

Collects per-reaction validation CSV files produced by the validator,
computes convergence rates, cycle counts, and energy statistics,
and saves a summary CSV.
"""

import argparse
from typing import List, Dict
import pandas as pd
import numpy as np
from motsart.common import PathHandler


def get_allowed_ts_files_per_rxn(rxn_ids_L: List, ts_method: str, validator: str, results_folder: str) -> Dict[int, set]:
    """Get the set of ts_file names present in the validation CSV for each reaction."""
    allowed = {}
    for rxn_id in rxn_ids_L:
        path_handler = PathHandler(rxn_id=rxn_id, r_or_p='r', ts_method=ts_method, validation_method=validator, results_folder=results_folder)
        res_csv_path = path_handler.validation_ts_method / f'validation_{validator}.csv'
        if res_csv_path.exists():
            df = pd.read_csv(res_csv_path)
            allowed[rxn_id] = set(df['ts_file'].tolist())
    return allowed


def compute_stats_for_ts_method(rxn_ids_L: List, ts_method: str, validator: str, cluster_folder: str, allowed_ts_files: Dict[int, set] = None) -> Dict:
    print(f"\nProcessing {ts_method} ---")
    df_L = []
    for rxn_id in rxn_ids_L:
        path_handler = PathHandler(rxn_id=rxn_id, r_or_p='r', ts_method=ts_method, validation_method=validator, results_folder=cluster_folder)
        res_csv_path = path_handler.validation_ts_method / f'validation_{validator}.csv'
        if not res_csv_path.exists():
            print(f"Skipping rxn {rxn_id}, not found: {res_csv_path}")
            continue

        df = pd.read_csv(res_csv_path)
        if allowed_ts_files is not None and rxn_id in allowed_ts_files:
            before = len(df)
            df = df[df['ts_file'].isin(allowed_ts_files[rxn_id])]
            print(f"  rxn {rxn_id}: filtered {before} -> {len(df)} molecules (matched with learning)")
        df_L.append(df)

    n_reactions = len(df_L)
    path_handler = PathHandler(ts_method=ts_method, results_folder=cluster_folder)
    df_total = pd.concat(df_L, ignore_index=True)
    n_molecules = len(df_total)
    df_total.to_csv(path_handler.validation_stats / f'validation_{ts_method}.csv', index=False)

    # Compute statistics
    converged_mean = df_total['converged'].mean()
    irc_converged_mean = df_total['irc_converged'].mean()

    # For rows where irc_converged is True
    irc_true_df = df_total[df_total['irc_converged'] == True]
    mean_cycle_cnt = irc_true_df['cycle_cnt'].mean() if len(irc_true_df) > 0 else np.nan
    median_cycle_cnt = irc_true_df['cycle_cnt'].median() if len(irc_true_df) > 0 else np.nan
    mean_energy = irc_true_df['energy_kcal'].mean() if len(irc_true_df) > 0 else np.nan
    median_energy = irc_true_df['energy_kcal'].median() if len(irc_true_df) > 0 else np.nan
    mad_energy = (irc_true_df['energy_kcal'] - irc_true_df['energy_kcal'].mean()).abs().mean() if not irc_true_df.empty else np.nan
    mad_cycle_cnt = (irc_true_df['cycle_cnt'] - irc_true_df['cycle_cnt'].mean()).abs().mean() if not irc_true_df.empty else np.nan

    print(f"  {ts_method}: {n_molecules} molecules across {n_reactions} reactions in final statistics")

    return {
        'ts_method': ts_method,
        'n_reactions': n_reactions,
        'n_molecules': n_molecules,
        'converged': f'{converged_mean:.3f}',
        'irc_converged': f'{irc_converged_mean:.3f}',
        'cycle_cnt_mean': f'{mean_cycle_cnt:.3f}' if not np.isnan(mean_cycle_cnt) else 'nan',
        'cycle_cnt_median': median_cycle_cnt if not np.isnan(median_cycle_cnt) else 'nan',
        'cycle_cnt_mad': f'{mad_cycle_cnt:.3f}' if not np.isnan(mad_cycle_cnt) else 'nan',
        'energy_kcal_mean': f'{mean_energy:.3f}' if not np.isnan(mean_energy) else 'nan',
        'energy_kcal_median': median_energy if not np.isnan(median_energy) else 'nan',
        'energy_kcal_mad': f'{mad_energy:.3f}' if not np.isnan(mad_energy) else 'nan'
    }

def collect_failures(rxn_ids_L: List, ts_method: str, validator: str, results_folder: str) -> pd.DataFrame:
    """Collect molecules where SP optimization or IRC did not succeed."""
    rows = []
    for rxn_id in rxn_ids_L:
        path_handler = PathHandler(rxn_id=rxn_id, r_or_p='r', ts_method=ts_method, validation_method=validator, results_folder=results_folder)
        res_csv_path = path_handler.validation_ts_method / f'validation_{validator}.csv'
        if not res_csv_path.exists():
            continue
        df = pd.read_csv(res_csv_path)
        sp_failed = df[df['converged'] == False]
        irc_failed = df[(df['converged'] == True) & (df['irc_converged'] == False)]
        ts_guess_dir = path_handler.ts_to_validate
        for _, row in sp_failed.iterrows():
            rows.append({'rxn_id': row['rxn_id'], 'ts_file': row['ts_file'], 'failure': 'sp_opt', 'path': str(ts_guess_dir / row['ts_file'])})
        for _, row in irc_failed.iterrows():
            rows.append({'rxn_id': row['rxn_id'], 'ts_file': row['ts_file'], 'failure': 'irc', 'path': str(ts_guess_dir / row['ts_file'])})
    return pd.DataFrame(rows)


def collect_all_results(rxn_ids_L: List, ts_method: str, validator: str, results_folder: str) -> pd.DataFrame:
    """Collect all per-molecule validation results into a single DataFrame."""
    df_L = []
    for rxn_id in rxn_ids_L:
        path_handler = PathHandler(rxn_id=rxn_id, r_or_p='r', ts_method=ts_method, validation_method=validator, results_folder=results_folder)
        res_csv_path = path_handler.validation_ts_method / f'validation_{validator}.csv'
        if not res_csv_path.exists():
            continue
        df_L.append(pd.read_csv(res_csv_path))
    if not df_L:
        return pd.DataFrame()
    return pd.concat(df_L, ignore_index=True)


def get_rxn_ids_from_al_folder(learning_folder: str) -> List[int]:
    """Get reaction IDs from folders in the learning directory (R0, R1, R2, etc.)"""
    from pathlib import Path
    al_path = Path(learning_folder)
    rxn_ids = []
    for folder in al_path.iterdir():
        if folder.is_dir() and folder.name.startswith('R'):
            try:
                rxn_id = int(folder.name[1:])
                rxn_ids.append(rxn_id)
            except ValueError:
                continue
    return sorted(rxn_ids)


def compute_and_save_stats(
    cluster_folder: str,
    learning_folder: str,
    validator_name: str,
    output_csv: str,
    cluster_ts_method: str,
    al_ts_method: str,
    mode: str = 'both'
):
    """
    Compute stats for TS methods.

    Args:
        cluster_folder: Path to cluster results folder
        learning_folder: Path to learning folder (used to determine which rxn IDs to process)
        validator: Validator name (e.g., 'GFN2XTBValidator')
        output_csv: Output CSV filename
        cluster_ts_method: TS method to validate for cluster results
        al_ts_method: TS method to validate for AL results
        mode: 'cluster', 'al', or 'both'
    """
    # Get reaction IDs from AL folder (this is the reference)
    rxn_ids_L = get_rxn_ids_from_al_folder(learning_folder)
    print(f"Found {len(rxn_ids_L)} reactions in AL folder: {rxn_ids_L}")

    all_stats = []

    # Build per-reaction set of mol files present in the learning folder
    # so we can restrict cluster stats to the same molecules
    al_ts_files = get_allowed_ts_files_per_rxn(rxn_ids_L, al_ts_method, validator_name, learning_folder)
    total_al_mols = sum(len(v) for v in al_ts_files.values())
    print(f"Learning folder has {total_al_mols} total molecules across {len(al_ts_files)} reactions")

    if mode in ('cluster', 'both'):
        print("\n=== Computing stats for CLUSTER results (filtered to learning molecules) ===")
        stats_dict = compute_stats_for_ts_method(
            rxn_ids_L, cluster_ts_method, validator_name,
            cluster_folder=cluster_folder,
            allowed_ts_files=al_ts_files
        )
        stats_dict['source'] = 'cluster'
        all_stats.append(stats_dict)

    if mode in ('al', 'both'):
        print("\n=== Computing stats for LEARNING results ===")
        stats_dict = compute_stats_for_ts_method(
            rxn_ids_L, al_ts_method, validator_name,
            cluster_folder=learning_folder
        )
        stats_dict['source'] = 'learning'
        all_stats.append(stats_dict)

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(output_csv, index=False)
    print(f"\nStats saved to: {output_csv}")

    # Save failures (non-filtered) for cluster and learning
    from pathlib import Path
    output_dir = Path(output_csv).parent

    if mode in ('cluster', 'both'):
        cluster_failures = collect_failures(rxn_ids_L, cluster_ts_method, validator_name, cluster_folder)
        cluster_fail_path = output_dir / f'failures_{cluster_ts_method}.csv'
        cluster_failures.to_csv(cluster_fail_path, index=False)
        print(f"Cluster failures ({len(cluster_failures)} molecules): {cluster_fail_path}")

    if mode in ('al', 'both'):
        learning_failures = collect_failures(rxn_ids_L, al_ts_method, validator_name, learning_folder)
        learning_fail_path = output_dir / f'failures_{al_ts_method}.csv'
        learning_failures.to_csv(learning_fail_path, index=False)
        print(f"Learning failures ({len(learning_failures)} molecules): {learning_fail_path}")

    # Save side-by-side comparison of cluster vs learning per molecule
    if mode == 'both':
        cluster_df = collect_all_results(rxn_ids_L, cluster_ts_method, validator_name, cluster_folder)
        learning_df = collect_all_results(rxn_ids_L, al_ts_method, validator_name, learning_folder)

        if not cluster_df.empty and not learning_df.empty:
            compare_cols = ['rxn_id', 'ts_file', 'converged', 'irc_converged', 'cycle_cnt', 'energy_kcal']
            cluster_df = cluster_df[[c for c in compare_cols if c in cluster_df.columns]]
            learning_df = learning_df[[c for c in compare_cols if c in learning_df.columns]]

            merged = pd.merge(
                cluster_df, learning_df,
                on=['rxn_id', 'ts_file'],
                suffixes=(f'_{cluster_ts_method}', f'_{al_ts_method}'),
                how='inner'
            )
            merged = merged.sort_values(['rxn_id', 'ts_file']).reset_index(drop=True)
            compare_path = output_dir / 'comparison.csv'
            merged.to_csv(compare_path, index=False)
            print(f"\nSide-by-side comparison ({len(merged)} molecules): {compare_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Compute validation statistics for TS methods')
    parser.add_argument('--cluster-folder', type=str, required=True,
                        help='Path to cluster results folder')
    parser.add_argument('--learning-folder', type=str, required=True,
                        help='Path to learning folder (determines which reactions to process)')
    parser.add_argument('--validator', type=str, default='GFN2XTBValidator',
                        help='Validator name (default: GFN2XTBValidator)')
    parser.add_argument('--output-csv', type=str, default='stats.csv',
                        help='Output CSV filename (default: stats.csv)')
    parser.add_argument('--cluster-ts-method', type=str, default='rmsd_pp',
                        help='TS method for cluster results (default: rmsd_pp)')
    parser.add_argument('--al-ts-method', type=str, default='goflow',
                        help='TS method for learning results (default: goflow)')
    parser.add_argument('--mode', type=str, choices=['cluster', 'al', 'both'], default='both',
                        help='Compute stats for: cluster, al (learning), or both (default: both)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_and_save_stats(
        cluster_folder=args.cluster_folder,
        learning_folder=args.learning_folder,
        validator_name=args.validator,
        output_csv=args.output_csv,
        cluster_ts_method=args.cluster_ts_method,
        al_ts_method=args.al_ts_method,
        mode=args.mode
    )
