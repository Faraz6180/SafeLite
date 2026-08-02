"""Runnable Meta-World demo for SafeLite.

This script creates a Meta-World environment, resets it, applies a short series
of random actions, prints observations and rewards, and closes the environment.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from simulator.environment import MetaWorldEnvironment


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_demo() -> None:
    """Run a simple random-action rollout for the Meta-World environment."""
    env = MetaWorldEnvironment(task_name="pick-place-v3", seed=42)

    try:
        env.initialize()
        reset_result = env.reset()
        logger.info("Initial observation: %s", reset_result["observation"])
        logger.info("Initial info: %s", reset_result["info"])

        for step_index in range(3):
            action: Any = [random.uniform(-1.0, 1.0) for _ in range(4)]
            step_result = env.step(action)
            logger.info("Step %d reward: %.4f", step_index + 1, step_result["reward"])
            logger.info("Step %d done: %s", step_index + 1, step_result["done"])
            logger.info("Step %d observation: %s", step_index + 1, step_result["observation"])

            if step_result["done"]:
                logger.info("Episode finished early at step %d.", step_index + 1)
                break
    except Exception as exc:  # pragma: no cover - defensive error handling
        logger.exception("Demo execution failed.")
        raise RuntimeError("Meta-World demo failed.") from exc
    finally:
        env.close()


if __name__ == "__main__":
    run_demo()
