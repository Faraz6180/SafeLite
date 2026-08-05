#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeLite Experiment Runner
Loop over tasks, seeds, and baselines. Log results to Parquet.
"""
import os
import sys
import yaml
import random
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pipeline import SafeLiteExecutionPipeline
from llm.factory import LLMProviderFactory
from simulator.environment import MetaWorldEnvironment
from executor.models import ExecutionResult

def load_config(config_path: str = "configs/experiment.yaml") -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_single_episode(task, seed, instruction, config, baseline_flags, llm_provider):
    random.seed(seed)
    simulator = MetaWorldEnvironment(task_name=task, render_mode=None, seed=seed)
    simulator.initialize()
    llm_config = config['llm']
    if llm_provider is None:
        llm = LLMProviderFactory.create_provider(provider_type=llm_config['provider'], model=llm_config['model'], temperature=llm_config['temperature'], max_tokens=llm_config['max_tokens'])
    else:
        llm = llm_provider
    pipeline = SafeLiteExecutionPipeline(llm_provider=llm, simulator=simulator, skip_safety=baseline_flags['skip_safety'], skip_correction=baseline_flags['skip_correction'], max_retries=3)
    start_time = time.perf_counter()
    try:
        result = pipeline.run(instruction)
        success = result.success
        steps_taken = result.steps_taken
        safety_violation = False
        error_message = result.error_message if not success else ""
    except Exception as e:
        success = False
        steps_taken = 0
        safety_violation = True
        error_message = str(e)
    finally:
        elapsed = time.perf_counter() - start_time
        simulator.close()
    return {
        "task": task,
        "seed": seed,
        "baseline": baseline_flags.get("baseline_name", "UNKNOWN"),
        "skip_safety": baseline_flags['skip_safety'],
        "skip_correction": baseline_flags['skip_correction'],
        "instruction": instruction,
        "success": success,
        "steps_taken": steps_taken,
        "safety_violation": safety_violation,
        "latency_seconds": elapsed,
        "error_message": error_message,
        "timestamp": pd.Timestamp.now()
    }

def main():
    config = load_config()
    exp_config = config['experiment']
    tasks = config['tasks']
    baseline_configs = config['baselines']
    num_seeds = exp_config['num_seeds']
    base_seed = exp_config['random_seed_base']
    raw_dir = Path(config['logging']['raw_data_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)
    instruction = "pick up the red block and place it on the green platform"
    all_results = []
    print(f"Starting SafeLite Experiment Suite")
    print(f"Tasks: {tasks}")
    print(f"Seeds: 0..{num_seeds-1}")
    print(f"Baselines: {list(baseline_configs.keys())}")
    print(f"Output: {raw_dir / 'experiment.parquet'}")
    print("-" * 60)
    total_runs = len(tasks) * num_seeds * len(baseline_configs)
    run_idx = 0
    llm_config = config['llm']
    llm_provider = LLMProviderFactory.create_provider(provider_type=llm_config['provider'], model=llm_config['model'], temperature=llm_config['temperature'], max_tokens=llm_config['max_tokens'])
    for task in tasks:
        for seed in range(num_seeds):
            effective_seed = base_seed + seed
            for baseline_name, baseline_flags in baseline_configs.items():
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] Task={task}, Seed={effective_seed}, Baseline={baseline_name}")
                flags_with_name = baseline_flags.copy()
                flags_with_name['baseline_name'] = baseline_name
                result_row = run_single_episode(task=task, seed=effective_seed, instruction=instruction, config=config, baseline_flags=flags_with_name, llm_provider=llm_provider)
                all_results.append(result_row)
                if run_idx % 10 == 0:
                    df = pd.DataFrame(all_results)
                    df.to_parquet(raw_dir / "experiment.parquet", index=False)
                    print(f"  Checkpoint saved ({len(all_results)} rows)")
    df = pd.DataFrame(all_results)
    df.to_parquet(raw_dir / "experiment.parquet", index=False)
    print("-" * 60)
    print(f"Experiment complete. {len(all_results)} runs saved to {raw_dir / 'experiment.parquet'}")
    print("Summary:")
    print(df.groupby(['task', 'baseline']).agg({'success': ['mean', 'count'], 'latency_seconds': ['mean', 'median']}))

if __name__ == "__main__":
    main()
