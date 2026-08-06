"""
Core evaluation metrics for SafeLite experiments.
Computes Success Rate (SR), Safety Violation Rate (SVR),
Plan Validity Rate (PVR), Latency, and Steps.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute all evaluation metrics from a Parquet DataFrame.

    Args:
        df: DataFrame with columns: task, baseline, success, safety_violation,
            latency_seconds, steps_taken, error_message

    Returns:
        Dictionary with metrics grouped by (task, baseline)
    """
    metrics = {}

    for (task, baseline), group in df.groupby(['task', 'baseline']):
        key = f"{task}_{baseline}"
        n = len(group)

        # Success Rate (SR)
        sr = group['success'].mean()

        # Safety Violation Rate (SVR)
        svr = group['safety_violation'].mean()

        # Plan Validity Rate (PVR) - error_message empty means plan was valid
        pvr = (group['error_message'] == '').mean()

        # Latency
        latency_mean = group['latency_seconds'].mean()
        latency_median = group['latency_seconds'].median()
        latency_std = group['latency_seconds'].std()

        # Steps taken
        steps_mean = group['steps_taken'].mean()
        steps_std = group['steps_taken'].std()

        metrics[key] = {
            'task': task,
            'baseline': baseline,
            'n': n,
            'success_rate': sr,
            'safety_violation_rate': svr,
            'plan_validity_rate': pvr,
            'latency_mean': latency_mean,
            'latency_median': latency_median,
            'latency_std': latency_std,
            'steps_mean': steps_mean,
            'steps_std': steps_std,
        }

    return metrics


def compute_bootstrap_ci(
    df: pd.DataFrame,
    metric_col: str = 'success',
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for a metric.

    Args:
        df: DataFrame with columns: task, baseline, metric_col
        metric_col: Column name to compute CI for (e.g., 'success')
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)

    Returns:
        Dictionary mapping (task_baseline) -> (lower_bound, upper_bound)
    """
    results = {}
    alpha = 1 - confidence

    for (task, baseline), group in df.groupby(['task', 'baseline']):
        key = f"{task}_{baseline}"
        values = group[metric_col].values

        if len(values) < 2:
            results[key] = (np.nan, np.nan)
            continue

        # Bootstrap
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        results[key] = (lower, upper)

    return results


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """
    Format metrics as a human-readable table.
    """
    lines = []
    lines.append(f"{'Task':<20} {'Baseline':<12} {'SR':<8} {'SVR':<8} {'PVR':<8} {'Latency (s)':<12} {'N':<5}")
    lines.append("-" * 85)

    for key, m in metrics.items():
        lines.append(
            f"{m['task']:<20} "
            f"{m['baseline']:<12} "
            f"{m['success_rate']:<8.3f} "
            f"{m['safety_violation_rate']:<8.3f} "
            f"{m['plan_validity_rate']:<8.3f} "
            f"{m['latency_mean']:<12.4f} "
            f"{m['n']:<5}"
        )

    return "\n".join(lines)


def save_metrics_to_csv(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Save metrics to a CSV file.
    """
    rows = []
    for key, m in metrics.items():
        rows.append({
            'task': m['task'],
            'baseline': m['baseline'],
            'n': m['n'],
            'success_rate': m['success_rate'],
            'safety_violation_rate': m['safety_violation_rate'],
            'plan_validity_rate': m['plan_validity_rate'],
            'latency_mean': m['latency_mean'],
            'latency_median': m['latency_median'],
            'latency_std': m['latency_std'],
            'steps_mean': m['steps_mean'],
            'steps_std': m['steps_std'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")