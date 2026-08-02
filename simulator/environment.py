"""Meta-World environment wrapper for SafeLite.

This module provides a small object-oriented interface for loading a Meta-World
environment, resetting it, stepping through it, and closing the simulator.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import metaworld
from gymnasium import Env


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MetaWorldEnvironment:
    """High-level wrapper around a Meta-World environment.

    Parameters
    ----------
    task_name: str
        The Meta-World task identifier to load. The installed Meta-World version
        expects V3 task names such as ``pick-place-v3``.
    seed: int | None
        Optional random seed for reproducible behavior.
    render_mode: str | None
        Optional rendering mode for the environment.
    """

    def __init__(self, task_name: str = "pick-place-v3", seed: int | None = None, render_mode: str | None = None) -> None:
        self.task_name = task_name
        self.seed = seed
        self.render_mode = render_mode
        self._env: Env | None = None
        self._ml: Any | None = None
        self._task: Any | None = None

    def initialize(self) -> None:
        """Initialize the Meta-World benchmark and load the requested environment."""
        try:
            logger.info("Initializing Meta-World benchmark for task '%s'.", self.task_name)
            self._ml = metaworld.ML1(self.task_name)
            env_name = self.task_name
            if env_name not in self._ml.train_classes:
                raise KeyError(f"Task '{self.task_name}' is not available in the loaded benchmark.")
            self._env = self._ml.train_classes[env_name]()
            if hasattr(self._ml, "train_tasks") and self._ml.train_tasks:
                self._task = self._ml.train_tasks[0]
                if hasattr(self._env, "set_task"):
                    self._env.set_task(self._task)
            logger.info("Environment '%s' created successfully.", self.task_name)
        except Exception as exc:  # pragma: no cover - defensive error handling
            logger.exception("Failed to initialize Meta-World environment '%s'.", self.task_name)
            raise RuntimeError(f"Unable to initialize Meta-World environment '{self.task_name}'.") from exc

    def reset(self) -> dict[str, Any]:
        """Reset the environment and return the initial observation."""
        if self._env is None:
            raise RuntimeError("Environment has not been initialized. Call initialize() first.")

        try:
            logger.info("Resetting environment '%s'.", self.task_name)
            if self.seed is not None:
                self._env.seed(self.seed)
                self._env.current_seed = self.seed
            if hasattr(self._env, "_freeze_rand_vec"):
                self._env._freeze_rand_vec = False
            if hasattr(self._env, "seeded_rand_vec"):
                self._env.seeded_rand_vec = True
            if hasattr(self._env, "reset"):
                observation, info = self._env.reset()
            else:
                observation, info = self._env.reset(seed=self.seed)
            logger.info("Environment reset complete for '%s'.", self.task_name)
            return {"observation": observation, "info": info}
        except Exception as exc:  # pragma: no cover - defensive error handling
            logger.exception("Reset failed for environment '%s'.", self.task_name)
            raise RuntimeError(f"Reset failed for environment '{self.task_name}'.") from exc

    def step(self, action: Any) -> dict[str, Any]:
        """Apply a single action and return the next observation, reward, done, and info."""
        if self._env is None:
            raise RuntimeError("Environment has not been initialized. Call initialize() first.")

        try:
            logger.info("Stepping environment '%s' with action '%s'.", self.task_name, action)
            if self._task is not None and hasattr(self._env, "set_task"):
                self._env.set_task(self._task)
            observation, reward, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            logger.info("Step completed for '%s' with reward %.4f and done=%s.", self.task_name, reward, done)
            return {
                "observation": observation,
                "reward": reward,
                "done": done,
                "info": info,
            }
        except Exception as exc:  # pragma: no cover - defensive error handling
            logger.exception("Stepping failed for environment '%s'.", self.task_name)
            raise RuntimeError(f"Step failed for environment '{self.task_name}'.") from exc

    def close(self) -> None:
        """Close the underlying environment if it exists."""
        if self._env is not None:
            try:
                logger.info("Closing environment '%s'.", self.task_name)
                self._env.close()
                logger.info("Environment '%s' closed successfully.", self.task_name)
            except Exception as exc:  # pragma: no cover - defensive error handling
                logger.exception("Error while closing environment '%s'.", self.task_name)
                raise RuntimeError(f"Failed to close environment '{self.task_name}'.") from exc
        else:
            logger.info("No environment to close for '%s'.", self.task_name)

    @property
    def env(self) -> Env | None:
        """Return the underlying Gymnasium environment instance."""
        return self._env
