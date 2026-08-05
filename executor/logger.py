"""
Execution Logger - Structured logging for experiments.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from executor.models import ExecutionResult


class ExecutionLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.raw_results = []

    def log_execution(self, result: ExecutionResult, metadata: Dict[str, Any]):
        row = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "success": result.success,
            "steps_taken": result.steps_taken,
            "error_message": result.error_message if not result.success else "",
            "plan_json": result.plan.json() if result.plan else None,
            "task": metadata.get("task", "unknown"),
            "seed": metadata.get("seed", -1),
            "baseline": metadata.get("baseline", "unknown"),
            "instruction": metadata.get("instruction", ""),
        }
        self.raw_results.append(row)

    def save_to_parquet(self, output_path: str = "results/raw/experiment.parquet"):
        if not self.raw_results:
            print("No results to save.")
            return
        df = pd.DataFrame(self.raw_results)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")