"""
Mock LLM provider for testing.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseLLMProvider, LLMProviderError
from .config import LLMConfig


class MockProvider(BaseLLMProvider):
    """Mock provider that returns a fixed plan."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate_plan(self, instruction: str) -> Dict[str, Any]:
        """Return a fixed mock plan."""
        return {
            "goal": instruction,
            "actions": [
                {"action_type": "move_to", "target_object": "red_block", "gripper": "open"},
                {"action_type": "pick", "target_object": "red_block", "gripper": "closed"},
                {"action_type": "move_to", "target_object": "green_platform", "gripper": "hold"},
                {"action_type": "place", "target_object": "green_platform", "gripper": "open"}
            ]
        }

    def validate_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the raw response."""
        if not isinstance(raw_response, dict):
            raise LLMProviderError("Response is not a dictionary.")
        if 'goal' not in raw_response:
            raise LLMProviderError("Missing 'goal' in response.")
        if 'actions' not in raw_response:
            raise LLMProviderError("Missing 'actions' in response.")
        if not isinstance(raw_response['actions'], list):
            raise LLMProviderError("'actions' must be a list.")
        return raw_response