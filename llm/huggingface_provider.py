"""
HuggingFace LLM provider implementation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .base import BaseLLMProvider, LLMProviderError
from .config import LLMConfig

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


class HuggingFaceProvider(BaseLLMProvider):
    """LLM provider using HuggingFace models (local or inference API)."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.pipeline = None
        if pipeline is not None:
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=config.model_name or "microsoft/phi-3-mini-4k-instruct",
                    device_map="auto",
                    max_new_tokens=config.max_tokens
                )
            except Exception as e:
                raise LLMProviderError(f"Failed to load HF model: {e}")

    def generate_plan(self, instruction: str) -> Dict[str, Any]:
        """Generate a plan using HuggingFace."""
        if self.pipeline is None:
            raise LLMProviderError("HuggingFace pipeline not initialized. Check dependencies.")

        prompt = self._build_prompt(instruction)

        try:
            response = self.pipeline(
                prompt,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_tokens,
                return_full_text=False
            )
            raw_text = response[0]['generated_text'].strip()
            # Extract JSON from the response (might contain extra text)
            parsed = json.loads(raw_text)
            return self.validate_response(parsed)
        except Exception as e:
            raise LLMProviderError(f"HF generation failed: {e}")

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

    def _build_prompt(self, instruction: str) -> str:
        """Build the prompt for HuggingFace."""
        return f"""
You are a robot planner for a Meta-World manipulation environment.

Given the instruction: "{instruction}"

Return a JSON object with:
- "goal": a string describing the overall goal.
- "actions": a list of objects, each with:
    - "action_type": one of ["move_to", "pick", "place", "open", "close", "push"]
    - "target_object": the name of the object (e.g., "red_block", "green_platform")
    - "gripper": one of ["open", "closed", "hold"] (optional)

Example:
{{
    "goal": "pick up the red block and place it on the green platform",
    "actions": [
        {{"action_type": "move_to", "target_object": "red_block", "gripper": "open"}},
        {{"action_type": "pick", "target_object": "red_block", "gripper": "closed"}},
        {{"action_type": "move_to", "target_object": "green_platform", "gripper": "hold"}},
        {{"action_type": "place", "target_object": "green_platform", "gripper": "open"}}
    ]
}}

Return only the JSON object, no extra text.
"""