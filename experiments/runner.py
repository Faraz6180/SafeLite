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
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# ===== FORCE .env LOADING =====
from dotenv import load_dotenv
load_dotenv()
# ==============================

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== IMPORTS =====
from pipeline.orchestrator import SafeLiteExecutionPipeline
from llm.factory import create_provider
from llm.config import LLMConfig
from simulator.environment import MetaWorldEnvironment
from executor.models import ExecutionResult
# ===================


def load_config(config_path: str = "configs/experiment.yaml") -> Dict[str, Any]:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_api_key() -> str | None:
    """
    Get the API key from environment variables.
    Tries GROQ_API_KEY, OPENROUTER_API_KEY, then HF_API_KEY.
    Returns None if none are set.
    """
    return (os.getenv("GROQ_API_KEY") or
            os.getenv("OPENROUTER_API_KEY") or
            os.getenv("HF_API_KEY"))


def run_single_episode(
    task: str,
    seed: int,
    instruction: str,
    config: Dict[str, Any],
    baseline_flags: Dict[str, bool],
    llm_provider
) -> Dict[str, Any]:
    """
    Run one episode with the given task, seed, and baseline configuration.
    Returns a dictionary of raw metrics.
    """
    # Set deterministic seed for reproducibility
    random.seed(seed)

    # 1. Instantiate simulator with the specific task
    simulator = MetaWorldEnvironment(
        task_name=task,
        render_mode=None,  # Headless for experiments
        seed=seed
    )
    simulator.initialize()

    # 2. Use the passed LLM provider (or create a fresh one if None)
    if llm_provider is None:
        llm_config_dict = config['llm']
        llm_config = LLMConfig(
            provider=llm_config_dict['provider'],
            model_name=llm_config_dict['model'],
            temperature=llm_config_dict['temperature'],
            max_tokens=llm_config_dict['max_tokens'],
            api_key=get_api_key(),
            timeout=30
        )
        llm = create_provider(llm_config)
    else:
        llm = llm_provider

    # 3. Instantiate pipeline with baseline flags
    pipeline = SafeLiteExecutionPipeline(
        llm_provider=llm,
        simulator=simulator,
        skip_safety=baseline_flags['skip_safety'],
        skip_correction=baseline_flags['skip_correction'],
        max_retries=3
    )

    # 4. Execute the pipeline
    start_time = time.perf_counter()
    try:
        result = pipeline.run(instruction)
        success = result.success
        steps_taken = result.steps_taken
        safety_violation = False
        error_message = result.error_message if not success else ""

        # ===== DEBUG: Print the plan =====
        if hasattr(result, 'plan') and result.plan is not None:
            print("\n" + "=" * 80)
            print("🔍 DEBUG: LLM GENERATED PLAN")
            print("=" * 80)
            if hasattr(result.plan, 'model_dump'):
                print(json.dumps(result.plan.model_dump(), indent=2))
            else:
                print(result.plan)
            print("=" * 80 + "\n")
        # =================================

    except Exception as e:
        success = False
        steps_taken = 0
        safety_violation = True
        error_message = str(e)
    finally:
        elapsed = time.perf_counter() - start_time
        simulator.close()

    # 5. Build raw result row
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
    """Main experiment loop."""
    config = load_config()
    exp_config = config['experiment']
    tasks = config['tasks']
    baseline_configs = config['baselines']
    num_seeds = exp_config['num_seeds']
    base_seed = exp_config['random_seed_base']

    # Create output directory
    raw_dir = Path(config['logging']['raw_data_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Fixed instruction for all episodes
    instruction = "pick up the red block and place it on the green platform"

    all_results: List[Dict[str, Any]] = []

    print(f"Starting SafeLite Experiment Suite")
    print(f"Tasks: {tasks}")
    print(f"Seeds: 0..{num_seeds-1}")
    print(f"Baselines: {list(baseline_configs.keys())}")
    print(f"Output: {raw_dir / 'experiment.parquet'}")
    print("-" * 60)

    total_runs = len(tasks) * num_seeds * len(baseline_configs)
    run_idx = 0

    # Create a single LLM provider to reuse across runs
    llm_config_dict = config['llm']
    llm_config = LLMConfig(
        provider=llm_config_dict['provider'],
        model_name=llm_config_dict['model'],
        temperature=llm_config_dict['temperature'],
        max_tokens=llm_config_dict['max_tokens'],
        api_key=get_api_key(),
        timeout=30
    )
    llm_provider = create_provider(llm_config)

    for task in tasks:
        for seed in range(num_seeds):
            effective_seed = base_seed + seed
            for baseline_name, baseline_flags in baseline_configs.items():
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] Task={task}, Seed={effective_seed}, Baseline={baseline_name}")

                flags_with_name = baseline_flags.copy()
                flags_with_name['baseline_name'] = baseline_name

                result_row = run_single_episode(
                    task=task,
                    seed=effective_seed,
                    instruction=instruction,
                    config=config,
                    baseline_flags=flags_with_name,
                    llm_provider=llm_provider
                )
                all_results.append(result_row)

                # Save checkpoint every 10 runs
                if run_idx % 10 == 0:
                    df = pd.DataFrame(all_results)
                    df.to_parquet(raw_dir / "experiment.parquet", index=False)
                    print(f"  Checkpoint saved ({len(all_results)} rows)")

    # Final save
    df = pd.DataFrame(all_results)
    df.to_parquet(raw_dir / "experiment.parquet", index=False)
    print("-" * 60)
    print(f"Experiment complete. {len(all_results)} runs saved to {raw_dir / 'experiment.parquet'}")
    print("Summary:")
    print(df.groupby(['task', 'baseline']).agg({
        'success': ['mean', 'count'],
        'latency_seconds': ['mean', 'median']
    }))


if __name__ == "__main__":
    main()