"""
Evaluation module for SafeLite.
Provides metrics computation, statistical analysis, and table generation.
"""
from .metrics import compute_metrics, format_metrics_table
from .compute_stats import main as compute_stats

__all__ = ["compute_metrics", "format_metrics_table", "compute_stats"]