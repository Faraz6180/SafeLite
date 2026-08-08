"""
Meta-World environment wrapper for SafeLite.
"""
import logging
import numpy as np
from typing import Dict, Any, Optional

import metaworld
from metaworld import MetaWorldEnv

from planner.models import Action

logger = logging.getLogger(__name__)


class MetaWorldEnvironment:
    """Wrapper for Meta-World environments."""

    def __init__(self, task_name: str = "pick-place-v3", render_mode: Optional[str] = None, seed: int = 42):
        self.task_name = task_name
        self.render_mode = render_mode
        self.seed = seed
        self._env = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        logger.info(f"Initializing Meta-World benchmark for task '{self.task_name}'.")

        ml1 = metaworld.ML1(self.task_name, seed=self.seed)
        self._env = ml1.train_classes[self.task_name](render_mode=self.render_mode)
        self._env.set_task(ml1.train_tasks[0])
        self._env.reset()
        self._env.seed(self.seed)
        self._env._freeze_rand_vec = False

        # If render_mode is "human", force a render to open the window
        if self.render_mode == "human":
            self._env.render()

        self._initialized = True
        logger.info(f"Environment '{self.task_name}' created successfully.")

    def reset(self) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()
        obs = self._env.reset()
        if self.render_mode == "human":
            self._env.render()
        return {"observation": obs}

    def step(self, action: Action) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        logger.info(f"Stepping environment '{self.task_name}' with action '{action}'.")

        low_level_action = np.zeros(4, dtype=np.float32)

        if hasattr(action, 'action_type'):
            if action.action_type == 'move_to':
                low_level_action = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)
            elif action.action_type == 'pick':
                low_level_action = np.array([0.0, -0.1, 0.0, 1.0], dtype=np.float32)
            elif action.action_type == 'place':
                low_level_action = np.array([0.0, 0.1, 0.0, -1.0], dtype=np.float32)

        logger.debug(f"Low-level action: {low_level_action}")

        try:
            observation, reward, terminated, truncated, info = self._env.step(low_level_action)
            # Force render to update the window
            if self.render_mode == "human":
                self._env.render()
            success = reward > 0.0
            return {
                "observation": observation,
                "reward": reward,
                "done": terminated or truncated,
                "success": success,
                "info": info
            }
        except Exception as e:
            logger.error(f"Stepping failed: {e}", exc_info=True)
            raise RuntimeError(f"Step failed: {e}") from e

    def close(self):
        if self._env:
            self._env.close()
            logger.info(f"Environment '{self.task_name}' closed successfully.")
            self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()