"""
OpenRouter LLM provider implementation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .base import BaseLLMProvider, LLMProviderError
from .config import LLMConfig

try:
    import requests
except ImportError:
    requests = None


class OpenRouterProvider(BaseLLMProvider):
    """LLM provider using OpenRouter API."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_plan(self, instruction: str) -> Dict[str, Any]:
        """Generate a plan using OpenRouter."""
        if not self.api_key:
            raise LLMProviderError("OpenRouter API key not set.")
        if requests is None:
            raise LLMProviderError("requests module not installed.")

        prompt = self._build_prompt(instruction)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model_name or "meta-llama/llama-3.1-8b-instruct",
            "messages": [
                {"role": "system", "content": "You are a robot planner. Return a JSON object with 'goal' and 'actions'."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            raw_text = data['choices'][0]['message']['content']
            parsed = json.loads(raw_text)
            return self.validate_response(parsed)
        except Exception as e:
            raise LLMProviderError(f"OpenRouter generation failed: {e}")

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
        """Build the prompt for OpenRouter."""
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