"""Validation rules for planner-generated action plans.

The validator enforces constraints on the structure and semantics of an
ActionPlan before it is accepted by the planner agent.
"""

from __future__ import annotations

from .models import ActionPlan


class PlanValidationError(ValueError):
    """Raised when a plan violates the planner validation rules."""


class PlanValidator:
    """Validate the semantics of an action plan."""

    VALID_ACTION_TYPES = {"pick", "place", "move", "open", "close", "hold"}
    VALID_GRIPPERS = {"open", "close", "hold"}

    @classmethod
    def validate(cls, plan: ActionPlan) -> None:
        """Validate a plan instance.

        Parameters
        ----------
        plan: ActionPlan
            The plan to validate.

        Raises
        ------
        PlanValidationError
            If the plan violates any validation rule.
        """
        if not plan.goal or not plan.goal.strip():
            raise PlanValidationError("Plan goal cannot be empty.")

        if not plan.actions:
            raise PlanValidationError("Plan cannot contain no actions.")

        seen_steps: set[tuple[str, str]] = set()
        for index, action in enumerate(plan.actions):
            if action.action_type not in cls.VALID_ACTION_TYPES:
                raise PlanValidationError(
                    f"Unknown action type at step {index + 1}: {action.action_type}"
                )

            if action.gripper not in cls.VALID_GRIPPERS:
                raise PlanValidationError(
                    f"Invalid gripper command at step {index + 1}: {action.gripper}"
                )

            if not action.target_object or not action.target_object.strip():
                raise PlanValidationError(
                    f"Invalid object reference at step {index + 1}: target_object is empty."
                )

            if not action.target_location or not action.target_location.strip():
                raise PlanValidationError(
                    f"Invalid location reference at step {index + 1}: target_location is empty."
                )

            step_key = (action.action_type, action.target_object.lower())
            if step_key in seen_steps:
                raise PlanValidationError(
                    f"Duplicate step detected at step {index + 1}: {action.action_type} {action.target_object}"
                )
            seen_steps.add(step_key)
