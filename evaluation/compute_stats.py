#!/usr/bin/env python3
"""
Compute statistics from experiment Parquet data.
Loads results/raw/experiment.parquet, computes metrics, bootstrap CIs,
and prints summary tables.
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import compute_metrics, format_metrics_table, compute_bootstrap_ci


def load_experiment_data(data_path: str = "results/raw/experiment.parquet") -> pd.DataFrame:
    """Load experiment data from Parquet file."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment data not found at {data_path}. Run experiments/runner.py first.")
    return pd.read_parquet(path)


def compute_wilcoxon_comparisons(df: pd.DataFrame, baseline_a: str = 'B0', baseline_b: str = 'B3') -> Dict[str, Any]:
    """
    Compute Wilcoxon signed-rank test between two baselines.
    Groups by task and compares success rates.
    """
    from scipy import stats
    results = {}

    for task in df['task'].unique():
        a_data = df[(df['task'] == task) & (df['baseline'] == baseline_a)]['success'].values
        b_data = df[(df['task'] == task) & (df['baseline'] == baseline_b)]['success'].values

        if len(a_data) > 1 and len(b_data) > 1 and len(a_data) == len(b_data):
            # Pair by index (assuming seeds are aligned)
            stat, p_value = stats.wilcoxon(a_data, b_data, alternative='two-sided')
            results[task] = {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'n': len(a_data)
            }
        else:
            results[task] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'n': min(len(a_data), len(b_data))
            }

    return results


def main():
    """Main entry point for statistics computation."""
    print("=" * 60)
    print("SafeLite Experiment Statistics")
    print("=" * 60)

    # Load data
    try:
        df = load_experiment_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run experiments/runner.py first to generate data.")
        return

    print(f"\nLoaded {len(df)} experiment runs.")
    print(f"Tasks: {df['task'].unique().tolist()}")
    print(f"Baselines: {df['baseline'].unique().tolist()}")
    print(f"Seeds: {df['seed'].nunique()}")
    print("-" * 60)

    # Compute metrics
    metrics = compute_metrics(df)
    print("\n=== METRICS SUMMARY ===")
    print(format_metrics_table(metrics))

    # Compute bootstrap CIs for success rate
    print("\n=== BOOTSTRAP 95% CONFIDENCE INTERVALS (Success Rate) ===")
    cis = compute_bootstrap_ci(df, 'success', n_bootstrap=1000)
    for key, (lower, upper) in cis.items():
        print(f"{key:<30}: {lower:.3f} - {upper:.3f}")

    # Wilcoxon comparisons (B0 vs B3) if both exist
    if 'B0' in df['baseline'].unique() and 'B3' in df['baseline'].unique():
        print("\n=== WILCOXON SIGNED-RANK TEST (B0 vs B3) ===")
        try:
            from scipy import stats
            comparisons = compute_wilcoxon_comparisons(df, 'B0', 'B3')
            for task, result in comparisons.items():
                sig = "✅ SIGNIFICANT" if result['significant'] else "❌ NOT significant"
                print(f"{task:<20}: p={result['p_value']:.4f} (n={result['n']}) {sig}")
        except ImportError:
            print("scipy not installed. Install with: pip install scipy")
    else:
        print("\n=== WILCOXON TEST SKIPPED ===")
        print("Both B0 and B3 baselines are required for comparison.")

    # Save metrics to CSV
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    from evaluation.metrics import save_metrics_to_csv
    save_metrics_to_csv(metrics, str(output_dir / "metrics_summary.csv"))

    print("\n" + "=" * 60)
    print("Statistics complete. Results saved to results/metrics_summary.csv")


if __name__ == "__main__":
    main()