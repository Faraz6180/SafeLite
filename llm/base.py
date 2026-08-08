"""
Base LLM provider abstraction for SafeLite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# ===== FIX: Remove top-level import from planner.models =====
# from planner.models import ActionPlan  # <-- DELETE THIS LINE


class LLMProviderError(Exception):
    """Raised when an LLM provider operation fails."""
    pass


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_plan(self, instruction: str) -> Dict[str, Any]:
        """
        Generate a structured plan from a natural-language instruction.

        Args:
            instruction: Natural-language task description.

        Returns:
            Dictionary containing the plan (goal, actions, etc.).
        """
        pass

    @abstractmethod
    def validate_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize a raw LLM response into a structured plan.

        Args:
            raw_response: Raw dictionary from the LLM.

        Returns:
            Validated plan dictionary.

        Raises:
            LLMProviderError: If validation fails.
        """
        pass

    def generate_and_validate(self, instruction: str) -> Dict[str, Any]:
        """
        Generate and validate a plan in one step.
        """
        raw = self.generate_plan(instruction)
        return self.validate_response(raw)

    # ===== FIX: Lazy import inside the method =====
    def parse_to_action_plan(self, raw_response: Dict[str, Any]) -> Any:
        """
        Parse a raw response into an ActionPlan object.
        Lazy import avoids circular dependency.
        """
        # Import locally to break circular dependency
        from planner.models import ActionPlan  # <-- MOVED HERE

        # Validate the response
        validated = self.validate_response(raw_response)

        # Convert to ActionPlan
        actions = validated.get('actions', [])
        goal = validated.get('goal', '')

        # Ensure actions are properly structured
        action_objects = []
        for act in actions:
            if isinstance(act, dict):
                action_objects.append(act)

        return ActionPlan(
            goal=goal,
            actions=action_objects,
            raw_response=raw_response
        )