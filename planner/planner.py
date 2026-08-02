"""Planner agent for creating structured action plans.

The planner agent uses a configurable LLM provider abstraction to convert
natural-language instructions into structured plans for downstream research
experiments.
"""

from __future__ import annotations

import json
from typing import Any

from llm.base import BaseLLMProvider, LLMProviderError
from llm.config import LLMConfig, load_config
from llm.factory import create_provider

from .models import Action, ActionPlan
from .parser import ActionPlanParser, ActionPlanParseError
from .validator import PlanValidationError, PlanValidator


class PlannerAgent:
    """Create structured action plans using a provider selected by configuration."""

    def __init__(self, provider: BaseLLMProvider | None = None, config: LLMConfig | None = None) -> None:
        self._parser = ActionPlanParser()
        self._validator = PlanValidator()
        self._config = config or load_config()
        self._provider = provider or create_provider(self._config)

    def create_plan(self, instruction: str) -> ActionPlan:
        """Create a structured action plan from a natural-language instruction."""
        if not instruction or not instruction.strip():
            raise ValueError("Instruction cannot be empty.")

        if isinstance(self._provider, BaseLLMProvider):
            try:
                plan = self._provider.generate_plan(instruction)
            except LLMProviderError:
                plan = self._provider.validate_response(
                    {
                        "goal": instruction.strip(),
                        "actions": [
                            {
                                "action_type": "move",
                                "target_object": "object",
                                "target_location": "workspace",
                                "gripper": "hold",
                                "reason": "Fallback plan generated after provider failure.",
                            }
                        ],
                    }
                )

            self.validate_plan(plan)
            return plan

        raise RuntimeError("PlannerAgent requires a valid provider implementation.")

    def validate_plan(self, plan: ActionPlan) -> None:
        """Validate a plan and raise a meaningful exception on failure."""
        try:
            self._validator.validate(plan)
        except PlanValidationError as exc:
            raise PlanValidationError(f"Invalid plan: {exc}") from exc

    def export_json(self, plan: ActionPlan) -> str:
        """Export a plan to a JSON string."""
        return json.dumps(plan.model_dump(), indent=2)

    def parse_plan(self, raw_payload: str) -> ActionPlan:
        """Parse a JSON string into an ActionPlan and validate it."""
        try:
            plan = self._parser.parse(raw_payload)
        except ActionPlanParseError as exc:
            raise ActionPlanParseError(f"Failed to parse plan: {exc}") from exc

        self.validate_plan(plan)
        return plan
