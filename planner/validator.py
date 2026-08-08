"""
Plan validator for SafeLite.
Validates the structure and semantics of an ActionPlan.
"""
from typing import List, Set
from planner.models import ActionPlan, Action


class PlanValidationError(Exception):
    """Raised when a plan fails validation."""
    pass


class PlanValidator:
    """Validate an ActionPlan for correctness and safety."""

    # Define allowed action types
    ALLOWED_ACTION_TYPES = {"move_to", "pick", "place", "open", "close", "push", "move"}

    def validate(self, plan: ActionPlan) -> None:
        """
        Validate the plan.

        Args:
            plan: ActionPlan to validate.

        Raises:
            PlanValidationError: If the plan is invalid.
        """
        if not plan.actions:
            raise PlanValidationError("Plan has no actions.")

        for idx, action in enumerate(plan.actions):
            # Check action type
            if action.action_type not in self.ALLOWED_ACTION_TYPES:
                raise PlanValidationError(
                    f"Unknown action type at step {idx + 1}: {action.action_type}"
                )

            # Check that target_object is present
            if not action.target_object:
                raise PlanValidationError(
                    f"Missing target_object at step {idx + 1}"
                )

            # Additional checks can be added here (e.g., object exists in scene)