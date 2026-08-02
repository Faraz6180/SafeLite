"""Validation logic for safety rules.

The validator checks single actions and complete plans against the configured
safety rules and raises meaningful exceptions when a violation is detected.
"""

from __future__ import annotations

import logging
from typing import List

from planner.models import Action, ActionPlan

from .exceptions import (
    InvalidActionSequenceError,
    InvalidWorkspaceError,
    SafetyViolationError,
    UnknownObjectError,
)
from .rules import SafetyRules

logger = logging.getLogger(__name__)


class SafetyValidator:
    """Validate actions and plans against reusable safety rules."""

    def __init__(self, rules: SafetyRules | None = None) -> None:
        self.rules = rules or SafetyRules()

    def validate_action(self, action: Action) -> None:
        """Validate a single action against the current safety rules."""
        if not action.action_type or not action.action_type.strip():
            raise SafetyViolationError("Action type cannot be empty.")

        if action.action_type.lower() not in self.rules.allowed_actions:
            raise SafetyViolationError(f"Unknown action type: {action.action_type}")

        if action.gripper.lower() not in self.rules.allowed_grippers:
            raise SafetyViolationError(f"Invalid gripper state: {action.gripper}")

        normalized_object = action.target_object.lower().strip()
        if normalized_object not in self.rules.allowed_objects:
            raise UnknownObjectError(f"Unknown object: {action.target_object}")

        if normalized_object in self.rules.forbidden_objects:
            raise SafetyViolationError(f"Forbidden object: {action.target_object}")

        normalized_location = action.target_location.lower().strip()
        if normalized_location not in self.rules.allowed_locations:
            raise InvalidWorkspaceError(f"Unknown location: {action.target_location}")

        if normalized_location == "workspace":
            return

        if normalized_location == "table":
            return

        if normalized_location == "basket":
            return

        if normalized_location == "box":
            return

    def validate_plan(self, plan: ActionPlan) -> None:
        """Validate an entire action plan.

        Parameters
        ----------
        plan: ActionPlan
            The plan to validate.

        Raises
        ------
        SafetyViolationError
            If the plan is empty or contains invalid ordering.
        """
        if not plan.goal or not plan.goal.strip():
            raise SafetyViolationError("Empty plan: goal is missing.")

        if not plan.actions:
            raise SafetyViolationError("Empty plan: no actions were provided.")

        seen_action_types: set[str] = set()
        previous_action: Action | None = None
        for index, action in enumerate(plan.actions):
            self.validate_action(action)

            action_signature = (action.action_type.lower(), action.target_object.lower())
            if action_signature in seen_action_types:
                raise InvalidActionSequenceError(
                    f"Duplicate action at step {index + 1}: {action.action_type} {action.target_object}"
                )
            seen_action_types.add(action_signature)

            if previous_action is not None and action.action_type.lower() == previous_action.action_type.lower():
                raise InvalidActionSequenceError(
                    f"Invalid action order at step {index + 1}: repeated action type '{action.action_type}'"
                )
            previous_action = action

        logger.info("Plan validation passed for goal '%s'.", plan.goal)
