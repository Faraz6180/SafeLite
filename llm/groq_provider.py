"""
Groq LLM provider implementation.
"""
from __future__ import annotations

import json
import os
import httpx
from typing import Any, Dict, Optional

from .base import BaseLLMProvider, LLMProviderError
from .config import LLMConfig

try:
    from groq import Groq
except ImportError:
    Groq = None


class GroqProvider(BaseLLMProvider):
    """LLM provider using Groq API."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = None
        if Groq is not None and config.api_key:
            # Create a custom HTTP client with SSL verification disabled for development
            http_client = httpx.Client(verify=False)
            self.client = Groq(api_key=config.api_key, http_client=http_client)

    def generate_plan(self, instruction: str) -> Dict[str, Any]:
        if not self.client:
            raise LLMProviderError("Groq client not initialized. Check API key.")

        prompt = self._build_prompt(instruction)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name or "llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a robot planner. Return a JSON object with 'goal' and 'actions'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            raw_text = response.choices[0].message.content
            parsed = json.loads(raw_text)
            return self.validate_response(parsed)
        except Exception as e:
            raise LLMProviderError(f"Groq generation failed: {e}")

    def validate_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
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