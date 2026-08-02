"""Reflection engine for interpreting failures and proposing recovery."""

from __future__ import annotations

from .models import FailureReport, ReflectionReport, RecoveryDecision


class ReflectionEngine:
    """Analyze a failure report and recommend a recovery action."""

    def analyze_failure(self, failure: FailureReport) -> ReflectionReport:
        """Generate a human-readable reflection for a detected failure."""
        if failure.failure_type == "unknown_object":
            summary = "The planner referenced an object that is not recognized by the safety rules."
            why = "The object was not present in the allowed object set."
            recommended = "Replace the target object with an allowed object or terminate safely."
        elif failure.failure_type == "timeout":
            summary = "The action exceeded the expected execution window."
            why = "The simulator or environment did not complete the action in time."
            recommended = "Retry the current action once before terminating or replanning."
        elif failure.failure_type == "collision":
            summary = "The action likely caused a collision or unsafe interaction."
            why = "The environment feedback indicates a risky state."
            recommended = "Terminate safely and request human intervention."
        elif failure.failure_type == "invalid_plan":
            summary = "The plan structure or ordering appears invalid."
            why = "The action sequence violates the validation rules."
            recommended = "Replan the action sequence before retrying."
        else:
            summary = "A non-specific execution failure was detected."
            why = "The underlying cause was not fully identified."
            recommended = "Retry once, then terminate if the issue persists."

        return ReflectionReport(
            summary=summary,
            why_failed=why,
            which_step=f"Step {failure.step_index + 1}",
            recommended_action=recommended,
        )

    def generate_reflection(self, failure: FailureReport) -> ReflectionReport:
        """Alias for analyze_failure for API consistency."""
        return self.analyze_failure(failure)

    def recommend_recovery(self, failure: FailureReport) -> RecoveryDecision:
        """Recommend a recovery decision based on the failure report."""
        if failure.failure_type in {"collision", "invalid_location", "unknown_object"}:
            return RecoveryDecision(
                strategy="terminate_safely",
                recoverable=False,
                reason="The failure is unsafe or semantically invalid.",
                confidence=failure.confidence,
                should_terminate=True,
            )
        if failure.failure_type == "timeout":
            return RecoveryDecision(
                strategy="retry_current_action",
                recoverable=True,
                reason="The action may succeed on retry.",
                confidence=failure.confidence,
            )
        if failure.failure_type == "invalid_plan":
            return RecoveryDecision(
                strategy="replan",
                recoverable=True,
                reason="The plan structure needs correction.",
                confidence=failure.confidence,
                should_replan=True,
            )
        return RecoveryDecision(
            strategy="retry_current_action",
            recoverable=True,
            reason="The failure is ambiguous; a cautious retry is recommended.",
            confidence=failure.confidence,
        )
