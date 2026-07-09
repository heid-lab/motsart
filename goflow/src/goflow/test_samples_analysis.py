import pickle 
from pathlib import Path
import random
import statistics
import pandas as pd
from tqdm import tqdm
import argparse
from typing import Optional

from goflow.gotennet.models.components.callbacks import evaluate_geometry

# Global configuration for metrics we want to track
# Note: mean_force_norm and max_force_norm are optional (only computed if evaluate_force_norm=True)
METRICS = ['mae', 'rmse', 'angle_error', 'dihedral_error', 'steric_clash', 'mean_force_norm', 'max_force_norm']


def _apply_force_eval_config(
    data,
    *,
    evaluate_force_norm: bool,
    force_backend: str,
    force_multiplicity: int,
    force_solvent: Optional[str],
) -> None:
    data.evaluate_force_norm = bool(evaluate_force_norm)
    data.force_backend = str(force_backend)
    data.force_multiplicity = int(force_multiplicity)
    data.force_solvent = force_solvent


def evaluate_position_variant(
    data,
    pos_tensor,
    *,
    evaluate_force_norm: bool,
    force_backend: str,
    force_multiplicity: int,
    force_solvent: Optional[str],
):
    """
    Evaluates a specific position tensor (candidate) for a reaction.
    """
    # Clone data to ensure we don't mutate the original object during the loop
    data_cp = data.clone()
    data_cp.pos_gen = pos_tensor
    _apply_force_eval_config(
        data_cp,
        evaluate_force_norm=evaluate_force_norm,
        force_backend=force_backend,
        force_multiplicity=force_multiplicity,
        force_solvent=force_solvent,
    )

    # Assumes evaluate_geometry is available in the global scope
    return evaluate_geometry(data_cp)

def get_best_and_random_metrics(sample_results, random_seed):
    """
    Analyzes a list of sample results to find the 'Best' (minimum error)
    and a deterministic 'Random' sample.
    """
    stats = {}
    num_samples = len(sample_results)
    if num_samples == 0:
        return stats

    # 1. Random Selection
    if random_seed is not None:
        random.seed(random_seed)
    rand_idx = random.randrange(num_samples)
    res_rand = sample_results[rand_idx]

    # 2. Calculate Best and extract Random
    for m in METRICS:
        # Get list of values for this metric across all samples (e.g. all MAEs)
        values = [res[m] for res in sample_results if m in res]
        
        if values:
            stats[f'{m}_best'] = min(values)       # Best = Lowest Error/Clash
            stats[f'{m}_random'] = res_rand.get(m) # Random = Value at random index
            
    return stats

def process_single_reaction(
    data,
    *,
    evaluate_force_norm: bool,
    evaluate_force_norm_guess: bool,
    evaluate_force_norm_ground_truth: bool,
    force_backend: str,
    force_multiplicity: int,
    force_solvent: Optional[str],
):
    """
    Orchestrates the evaluation for a single reaction object.
    Computes Median, Best-of-Samples, Random-of-Samples, and Baseline(Guess).
    """
    # --- 1. Setup Data & Metadata ---
    if not hasattr(data, 'pos_gen_all_samples_S_N_3'):
        data.pos_gen_all_samples_S_N_3 = data.pos_gen.unsqueeze(0)
    
    row_stats = {
        'rxn_id': data.rxn_index.item() if hasattr(data.rxn_index, 'item') else data.rxn_index
    }

    _apply_force_eval_config(
        data,
        evaluate_force_norm=evaluate_force_norm,
        force_backend=force_backend,
        force_multiplicity=force_multiplicity,
        force_solvent=force_solvent,
    )

    # --- 2. Evaluate Median Geometry (stored in data.pos_gen) ---
    res_med = evaluate_geometry(data)
    for m in METRICS:
        row_stats[f'{m}_median'] = res_med.get(m)

    # --- 3. Evaluate All Generated Samples ---
    # Note: Force evaluation disabled for samples loop (only median/guess get forces)
    sample_results = []
    # Loop through the tensor of generated samples
    for i in range(len(data.pos_gen_all_samples_S_N_3)):
        res = evaluate_position_variant(
            data,
            data.pos_gen_all_samples_S_N_3[i],
            evaluate_force_norm=False,
            force_backend=force_backend,
            force_multiplicity=force_multiplicity,
            force_solvent=force_solvent,
        )
        sample_results.append(res)
    
    # Aggregate "Best" and "Random" from the list
    sample_stats = get_best_and_random_metrics(sample_results, random_seed=None)
    row_stats.update(sample_stats)

    # --- 4. Evaluate Filter-Selected Sample (if filters were applied) ---
    # Note: Force evaluation disabled for filter (only median/guess get forces)
    if hasattr(data, 'pos_gen_filter'):
        res_filter = evaluate_position_variant(
            data,
            data.pos_gen_filter,
            evaluate_force_norm=False,
            force_backend=force_backend,
            force_multiplicity=force_multiplicity,
            force_solvent=force_solvent,
        )
        for m in METRICS:
            row_stats[f'{m}_filter'] = res_filter.get(m)

    # --- 5. Evaluate Baseline (Guess) ---
    if hasattr(data, 'pos_guess'):
        res_guess = evaluate_position_variant(
            data,
            data.pos_guess,
            evaluate_force_norm=evaluate_force_norm_guess,
            force_backend=force_backend,
            force_multiplicity=force_multiplicity,
            force_solvent=force_solvent,
        )
        for m in METRICS:
            row_stats[f'{m}_guess'] = res_guess.get(m)

    # --- 6. Evaluate Ground Truth ---
    if hasattr(data, 'pos'):
        res_ground_truth = evaluate_position_variant(
            data,
            data.pos,
            evaluate_force_norm=evaluate_force_norm_ground_truth,
            force_backend=force_backend,
            force_multiplicity=force_multiplicity,
            force_solvent=force_solvent,
        )
        for m in METRICS:
            row_stats[f'{m}_ground_truth'] = res_ground_truth.get(m)

    return row_stats

def compute_summary_statistics(df):
    """
    Aggregates the per-reaction DataFrame into summary means for each category.
    """
    summary_rows = []
    categories = ['best', 'median', 'filter', 'random', 'guess', 'ground_truth']
    
    for cat in categories:
        # Find all columns belonging to this category (e.g. 'mae_best', 'rmse_best')
        cols = [c for c in df.columns if c.endswith(f'_{cat}')]
        
        if not cols:
            continue
            
        # Compute mean across all reactions
        mean_stats = df[cols].mean()
        
        # Build summary dict
        res = {'category': f'{cat}_of_all', 'n': len(df)}
        for col in cols:
            # Convert "mae_best" -> "mean_mae"
            metric_base = col.replace(f'_{cat}', '')
            res[f'mean_{metric_base}'] = mean_stats[col]
            
        summary_rows.append(res)
        
    return pd.DataFrame(summary_rows)

def save_analysis_results(per_rxn_df, summary_df, out_path):
    """
    Handles saving CSVs to disk and printing summary to console.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save Per-Reaction Stats
    per_rxn_df.sort_values('rxn_id').to_csv(out_path / 'stats_per_reaction.csv', index=False, float_format='%.6f')
    
    # 2. Save and Print Means
    summary_df.to_csv(out_path / 'summary_means.csv', index=False, float_format='%.6f')
    print("\nMean Summary (All Metrics):")
    print(summary_df.to_string(index=False))

def load_samples(samples_path: Path):
    """
    Loads the pickled samples list from disk.
    """
    print(f"Loading samples from {samples_path}...")
    with open(samples_path, 'rb') as f:
        samples_all = pickle.load(f)
    return samples_all

def save_samples_list(samples_L, out_path):
    pickle_save_path = out_path / 'samples_all.pkl'
    pickle_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_save_path, "wb") as f:
        pickle.dump(samples_L, f)
    print(f"Saved samples for head to {pickle_save_path.resolve()}")


def process_and_save_stats(
    samples_list,
    out_path,
    *,
    evaluate_force_norm: bool = False,
    evaluate_force_norm_guess: bool = False,
    evaluate_force_norm_ground_truth: bool = False,
    force_backend: str = "xtb",
    force_multiplicity: int = 1,
    force_solvent: Optional[str] = None,
):
    """
    Takes a list of data samples, computes statistics, and saves the results.
    """
    all_rows = []
    
    print("Processing reactions...")
    for data in tqdm(samples_list):
        try:
            row = process_single_reaction(
                data,
                evaluate_force_norm=evaluate_force_norm,
                evaluate_force_norm_guess=evaluate_force_norm_guess,
                evaluate_force_norm_ground_truth=evaluate_force_norm_ground_truth,
                force_backend=force_backend,
                force_multiplicity=force_multiplicity,
                force_solvent=force_solvent,
            )
        except Exception as e:
            print(f"Failed rxn {data.rxn_index}: {e}")
            continue
        all_rows.append(row)

    # Aggregate
    per_rxn_df = pd.DataFrame(all_rows)
    summary_df = compute_summary_statistics(per_rxn_df)
    
    # Save
    save_samples_list(samples_list, out_path)
    save_analysis_results(per_rxn_df, summary_df, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute stats for saved samples file")
    parser.add_argument("samples_path", nargs='?', default='xxx', help='Path to the pickled samples file')
    parser.add_argument("out_path", nargs='?', default='xxx', help='Output folder under reaction_analysis')
    parser.add_argument(
        "--evaluate-force-norm",
        action="store_true",
        help="Enable force-norm metrics (mean/max) via compute_forces backend",
    )
    parser.add_argument(
        "--force-backend",
        type=str,
        default="xtb",
        choices=["xtb"],
        help="Force backend used when --evaluate-force-norm is enabled",
    )
    parser.add_argument(
        "--force-multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity used for force calculation (default: 1)",
    )
    parser.add_argument(
        "--evaluate-force-norm-guess",
        action="store_true",
        help="Enable force-norm metrics for pos_guess baseline evaluation",
    )
    parser.add_argument(
        "--evaluate-force-norm-ground-truth",
        action="store_true",
        help="Enable force-norm metrics for ground truth (pos) evaluation",
    )
    parser.add_argument(
        "--force-solvent",
        type=str,
        default=None,
        help="Solvent for implicit solvation in force calculation (e.g., 'water', 'thf', 'dmso'). Uses XTB --alpb flag.",
    )
    args = parser.parse_args()

    # 1. Load Data
    samples_data = load_samples(Path(args.samples_path))
    
    # 2. Process and Save
    process_and_save_stats(
        samples_data,
        Path(args.out_path),
        evaluate_force_norm=args.evaluate_force_norm,
        evaluate_force_norm_guess=args.evaluate_force_norm_guess,
        evaluate_force_norm_ground_truth=args.evaluate_force_norm_ground_truth,
        force_backend=args.force_backend,
        force_multiplicity=args.force_multiplicity,
        force_solvent=args.force_solvent,
    )