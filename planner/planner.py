""""Planner agent for creating structured action plans.

The planner agent uses a configurable LLM provider abstraction to convert
natural-language instructions into structured plans for downstream research
experiments.
"""

from __future__ import annotations

import json
from typing import Any

from llm.base import BaseLLMProvider, LLMProviderError
from llm.config import LLMConfig, load_config

from .models import Action, ActionPlan
from .parser import ActionPlanParser, ActionPlanParseError
from .validator import PlanValidationError, PlanValidator


class PlannerAgent:
    """Create structured action plans using a provider selected by configuration."""

    def __init__(self, provider: BaseLLMProvider | None = None, config: LLMConfig | None = None) -> None:
        self._parser = ActionPlanParser()
        self._validator = PlanValidator()
        self._config = config or load_config()
        self._provider = provider

    def create_plan(self, instruction: str) -> ActionPlan:
        """Create a structured action plan from a natural-language instruction."""
        if not instruction or not instruction.strip():
            raise ValueError("Instruction cannot be empty.")

        if self._provider is None:
            raise RuntimeError("PlannerAgent requires a valid provider implementation.")

        # Generate the plan
        raw_response = self._provider.generate_plan(instruction)

        # ===== DEBUG: Print the raw LLM response =====
        print("=" * 60)
        print("LLM RAW RESPONSE:")
        print(raw_response)
        print("=" * 60)
        # =============================================

        # Validate the plan
        validated = self._provider.validate_response(raw_response)

        # Parse into ActionPlan (import locally to avoid circular issues)
        from .models import ActionPlan
        actions_list = validated.get('actions', [])
        plan = ActionPlan(
            goal=validated.get('goal', ''),
            actions=actions_list,
            raw_response=raw_response
        )
        self.validate_plan(plan)
        return plan

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