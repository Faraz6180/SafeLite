"""Offline mock provider used for deterministic tests and local development."""

from __future__ import annotations

from typing import Any

from planner.models import Action, ActionPlan

from .base import BaseLLMProvider, LLMProviderError
from .config import LLMConfig


class MockProvider(BaseLLMProvider):
    """Return a deterministic plan without contacting any remote service."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__(config or LLMConfig(provider="mock"))

    def generate_plan(self, instruction: str) -> ActionPlan:
        if not instruction or not instruction.strip():
            raise LLMProviderError("Instruction cannot be empty.")

        lowered = instruction.strip().lower()
        if "pick up" in lowered and "place" in lowered:
            actions = [
                Action(action_type="pick", target_object="object", target_location="workspace", gripper="close", reason="Pick up the referenced object."),
                Action(action_type="place", target_object="object", target_location="target location", gripper="open", reason="Place the object at the requested destination."),
            ]
        else:
            actions = [
                Action(action_type="move", target_object="object", target_location="workspace", gripper="hold", reason="Move toward the requested target."),
            ]

        return ActionPlan(goal=instruction.strip(), actions=actions)

    def health_check(self) -> bool:
        return True

    def validate_response(self, payload: Any) -> ActionPlan:
        if isinstance(payload, ActionPlan):
            return payload
        if isinstance(payload, dict):
            return ActionPlan.model_validate(payload)
        raise LLMProviderError("Mock provider only accepts an ActionPlan or dictionary payload.")
