"""Deterministic replanning logic for recovery after failures."""

from __future__ import annotations

from planner.models import Action, ActionPlan

from .models import FailureReport, ReplannedAction, RecoveryStrategy
from .exceptions import SelfCorrectionError


class Replanner:
    """Create a corrected action plan from a failure report."""

    def __init__(self, fallback_object: str = "object") -> None:
        self.fallback_object = fallback_object

    def replan(self, failure: FailureReport, current_plan: ActionPlan | None = None) -> list[ReplannedAction]:
        """Create a deterministic replacement action list."""
        if failure.failure_type == "invalid_plan" and current_plan is None:
            raise SelfCorrectionError("A current plan is required for invalid plan recovery")

        if failure.failure_type == "collision":
            return [
                ReplannedAction(
                    action_type="abort",
                    target_object=self.fallback_object,
                    target_location="safe_zone",
                    gripper="open",
                    reason="Unsafe collision detected; stop and isolate the task.",
                )
            ]

        if failure.failure_type == "timeout":
            return [
                ReplannedAction(
                    action_type="retry",
                    target_object=current_plan.actions[-1].target_object if current_plan else self.fallback_object,
                    target_location=current_plan.actions[-1].target_location if current_plan else "default",
                    gripper="close",
                    reason="Retry the last action with a conservative adjustment.",
                )
            ]

        if failure.failure_type == "unknown_object":
            return [
                ReplannedAction(
                    action_type="replace_target",
                    target_object=self.fallback_object,
                    target_location="default",
                    gripper="open",
                    reason="The original target object is not recognized.",
                )
            ]

        return [
            ReplannedAction(
                action_type="pause",
                target_object=self.fallback_object,
                target_location="default",
                gripper="open",
                reason="Unable to infer a safe recovery action; pause execution.",
            )
        ]

    def build_strategy(self, failure: FailureReport) -> RecoveryStrategy:
        """Return a high-level recovery strategy."""
        if failure.failure_type == "collision":
            return RecoveryStrategy(
                name="terminate_safely",
                description="Stop the task immediately and await intervention.",
                recoverable=False,
                replan_required=False,
            )
        if failure.failure_type == "timeout":
            return RecoveryStrategy(
                name="retry_once",
                description="Retry the same action once with more conservative settings.",
                recoverable=True,
                replan_required=False,
            )
        if failure.failure_type == "invalid_plan":
            return RecoveryStrategy(
                name="replan",
                description="Construct a corrected sequence of actions.",
                recoverable=True,
                replan_required=True,
            )
        return RecoveryStrategy(
            name="defer",
            description="Pause and defer the task for a safer later attempt.",
            recoverable=True,
            replan_required=True,
        )
