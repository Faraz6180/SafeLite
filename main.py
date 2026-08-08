#!/usr/bin/env python3
"""
SafeLite: Lightweight Language‑Conditioned Robot Manipulation

This is the main entry point for running the SafeLite pipeline interactively
with a live simulation window. It demonstrates the complete system:
LLM planning → Safety verification → Execution → Self‑correction (optional).

Usage:
    python main.py [--instruction "pick up the red block"]
                   [--task pick-place-v3] [--seed 42]
                   [--no-render] [--debug]

Example:
    python main.py --instruction "push the red block to the left"
                  --task push-v3 --seed 123

Author: Faraz Mubeen Haider
Paper: SafeLite: A Lightweight Open‑Source LLM Agent for Verifiable
       Language‑Conditioned Robotic Manipulation
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

# Add project root to path if needed
# sys.path.insert(0, ".")

from llm.config import load_config
from llm.factory import create_provider
from pipeline.orchestrator import SafeLiteExecutionPipeline
from simulator.environment import MetaWorldEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the SafeLite pipeline with live simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive prompt (default)
  python main.py

  # Run a specific instruction with rendering
  python main.py --instruction "push the red block to the left" --task push-v3

  # Headless (no GUI) – faster for experiments
  python main.py --instruction "pick up the red block" --no-render

  # Debug mode with verbose logging
  python main.py --instruction "open the gripper" --debug
        """,
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Natural language instruction (if omitted, user is prompted).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick-place-v3",
        choices=["reach-v3", "push-v3", "pick-place-v3"],
        help="Meta‑World task environment (default: pick-place-v3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run without the simulation window (headless).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def get_instruction(args: argparse.Namespace) -> str:
    """Get instruction from args or interactive prompt."""
    if args.instruction:
        return args.instruction
    return input("\nEnter a natural language instruction: ").strip()


def main() -> None:
    """Run the SafeLite pipeline with live simulation."""
    args = parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    print("\n" + "=" * 70)
    print("  SafeLite: Lightweight Language‑Conditioned Robot Manipulation")
    print("=" * 70)
    print(f"  Task: {args.task} (seed={args.seed})")
    print(f"  Render: {'ON' if not args.no_render else 'OFF'}")
    print("-" * 70)

    # 1. Load LLM configuration from .env
    try:
        cfg = load_config()
        logger.info(f"LLM provider: {cfg.provider}")
        logger.info(f"Model: {cfg.model_name}")
    except Exception as e:
        logger.error(f"Failed to load LLM config: {e}")
        sys.exit(1)

    # 2. Create LLM provider
    try:
        provider = create_provider(cfg)
    except Exception as e:
        logger.error(f"Failed to create LLM provider: {e}")
        sys.exit(1)

    # 3. Create simulator (with or without rendering)
    render_mode = None if args.no_render else "human"
    try:
        sim = MetaWorldEnvironment(
            task_name=args.task,
            render_mode=render_mode,
            seed=args.seed,
        )
        sim.initialize()
        logger.info(f"Simulator '{args.task}' initialized (render={render_mode}).")
    except Exception as e:
        logger.error(f"Failed to initialize simulator: {e}")
        sys.exit(1)

    # 4. Get instruction
    instruction = get_instruction(args)
    if not instruction:
        instruction = "pick up the red block and place it on the green platform"
        logger.warning(f"Using default instruction: {instruction}")

    # 5. Create pipeline
    pipeline = SafeLiteExecutionPipeline(
        llm_provider=provider,
        simulator=sim,
        skip_safety=False,          # Safety guard enabled
        skip_correction=True,       # Disable self‑correction for clean demo
        max_retries=3,
    )

    # 6. Execute
    print(f"\n▶ Executing: \"{instruction}\"")
    print("  (Watch the simulation window...)")
    start_time = time.time()

    try:
        result = pipeline.run(instruction)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        sim.close()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}", exc_info=True)
        sim.close()
        sys.exit(1)

    elapsed = time.time() - start_time

    # 7. Print summary (publication‑ready format)
    print("\n" + "=" * 70)
    print("  EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Success:         {'✅ YES' if result.success else '❌ NO'}")
    print(f"  Completed steps: {len(result.completed_actions)}")
    print(f"  Execution time:  {elapsed:.2f} seconds")
    if result.errors:
        print(f"  Errors:          {', '.join(result.errors)}")
    print("-" * 70)
    if result.logs:
        print("  Logs (last 5 entries):")
        for log in result.logs[-5:]:
            print(f"    • {log}")
    print("=" * 70)

    # 8. Keep window open for a few seconds if rendering
    if not args.no_render:
        print("\nThe simulation window will close in 5 seconds...")
        time.sleep(5)

    sim.close()
    logger.info("Demo complete.")


if __name__ == "__main__":
    main()

    