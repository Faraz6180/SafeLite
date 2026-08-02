"""Parser utilities for converting raw JSON into structured action plans.

The parser validates that the JSON payload is well-formed and that it matches
expected schema constraints before constructing an ActionPlan.
"""

from __future__ import annotations

import json
from typing import Any

from .models import Action, ActionPlan


class ActionPlanParseError(ValueError):
    """Raised when a raw JSON payload cannot be parsed into an ActionPlan."""


class ActionPlanParser:
    """Parse and normalize raw JSON data into an ActionPlan instance."""

    @staticmethod
    def parse(raw_payload: str) -> ActionPlan:
        """Parse a JSON string into an ActionPlan.

        Parameters
        ----------
        raw_payload: str
            A JSON string containing the plan payload.

        Returns
        -------
        ActionPlan
            The parsed and validated plan.

        Raises
        ------
        ActionPlanParseError
            If the payload is malformed or missing required structure.
        """
        try:
            payload: Any = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise ActionPlanParseError("Malformed JSON: unable to decode the plan payload.") from exc

        if not isinstance(payload, dict):
            raise ActionPlanParseError("Malformed JSON: expected an object at the top level.")

        goal = payload.get("goal")
        actions_payload = payload.get("actions")

        if not isinstance(goal, str) or not goal.strip():
            raise ActionPlanParseError("Missing required field: 'goal' must be a non-empty string.")

        if not isinstance(actions_payload, list):
            raise ActionPlanParseError("Missing required field: 'actions' must be a list.")

        if not actions_payload:
            raise ActionPlanParseError("Missing required field: 'actions' cannot be empty.")

        normalized_actions: list[Action] = []
        for index, action_payload in enumerate(actions_payload):
            if not isinstance(action_payload, dict):
                raise ActionPlanParseError(f"Malformed action at index {index}: expected an object.")

            try:
                action = Action.model_validate(action_payload)
            except Exception as exc:  # pragma: no cover - defensive error handling
                raise ActionPlanParseError(
                    f"Malformed action at index {index}: {exc}"
                ) from exc
            normalized_actions.append(action)

        return ActionPlan(goal=goal.strip(), actions=normalized_actions)
