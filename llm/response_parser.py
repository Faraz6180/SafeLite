"""Parse and validate responses from remote LLM providers."""

from __future__ import annotations

import json
from typing import Any

from planner.models import ActionPlan

from .base import LLMProviderError


class ResponseParser:
    """Convert provider responses into validated ActionPlan instances."""

    def parse(self, raw_response: Any) -> ActionPlan:
        if isinstance(raw_response, ActionPlan):
            return raw_response

        if isinstance(raw_response, str):
            text = raw_response.strip()
            if not text:
                raise LLMProviderError("Empty response received from provider.")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMProviderError("Malformed JSON response from provider.") from exc
            return self._validate_payload(payload)

        if isinstance(raw_response, dict):
            return self._validate_payload(raw_response)

        raise LLMProviderError("Unsupported response format from provider.")

    def _validate_payload(self, payload: Any) -> ActionPlan:
        if not isinstance(payload, dict):
            raise LLMProviderError("Response payload must be a JSON object.")

        if "goal" not in payload or "actions" not in payload:
            raise LLMProviderError("Response payload is missing required fields: goal and actions.")

        try:
            plan = ActionPlan.model_validate(payload)
        except Exception as exc:
            raise LLMProviderError(f"Response validation failed: {exc}") from exc

        return plan
