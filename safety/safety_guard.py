"""
Safety guard for SafeLite.
Validates plans against safety rules.
"""
from typing import List, Set
from planner.models import ActionPlan
from safety.models import SafetyResult


class SafetyGuard:
    """Safety guard that checks plans against allowed actions and objects."""

    ALLOWED_ACTION_TYPES = {"move_to", "pick", "place", "open", "close", "push", "move"}

    def validate_plan(self, plan: ActionPlan) -> SafetyResult:
        """
        Validate a plan.

        Args:
            plan: ActionPlan to validate.

        Returns:
            SafetyResult indicating validity.
        """
        violations = []

        # Check each action
        for idx, action in enumerate(plan.actions):
            # Check action type
            if action.action_type not in self.ALLOWED_ACTION_TYPES:
                violations.append(f"Unknown action type at step {idx+1}: {action.action_type}")

            # Check that target_object is not empty
            if not action.target_object:
                violations.append(f"Missing target_object at step {idx+1}")

            # Add more checks as needed (e.g., object exists in scene)

        if violations:
            return SafetyResult(
                valid=False,
                reason="; ".join(violations),
                violations=violations
            )
        else:
            return SafetyResult(
                valid=True,
                reason="All actions are safe."
            )