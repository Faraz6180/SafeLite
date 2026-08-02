"""Failure detection and classification for execution outcomes."""

from __future__ import annotations

from typing import Any

from executor.models import ExecutionResult
from safety.models import SafetyResult

from .models import FailureReport, FailureType


class FailureDetector:
    """Analyze execution and safety results to infer the failure type."""

    def __init__(self) -> None:
        self._last_report: FailureReport | None = None

    def detect(self, execution_result: ExecutionResult, safety_result: SafetyResult | None = None, simulator_feedback: str | None = None) -> FailureReport:
        """Detect and classify the failure from execution and safety information."""
        errors = execution_result.errors or []
        message = " ".join(errors) if errors else "Unknown execution failure"

        if simulator_feedback and "collision" in simulator_feedback.lower():
            failure_type = FailureType.COLLISION
            severity = "high"
            recoverable = False
            confidence = 0.95
        elif any("timeout" in error.lower() for error in errors):
            failure_type = FailureType.TIMEOUT
            severity = "medium"
            recoverable = True
            confidence = 0.9
        elif any("unknown object" in error.lower() for error in errors):
            failure_type = FailureType.UNKNOWN_OBJECT
            severity = "high"
            recoverable = False
            confidence = 0.95
        elif any("unknown location" in error.lower() for error in errors) or (safety_result is not None and not safety_result.approved):
            failure_type = FailureType.INVALID_LOCATION
            severity = "high"
            recoverable = False
            confidence = 0.88
        elif any("invalid plan" in error.lower() for error in errors):
            failure_type = FailureType.INVALID_PLAN
            severity = "high"
            recoverable = True
            confidence = 0.92
        elif any("simulator" in error.lower() for error in errors):
            failure_type = FailureType.SIMULATOR_ERROR
            severity = "medium"
            recoverable = True
            confidence = 0.85
        elif any("unexpected" in error.lower() for error in errors):
            failure_type = FailureType.UNKNOWN_ERROR
            severity = "medium"
            recoverable = True
            confidence = 0.75
        else:
            failure_type = FailureType.UNKNOWN_ERROR
            severity = "medium"
            recoverable = True
            confidence = 0.6

        report = FailureReport(
            failure_type=failure_type,
            severity=severity,
            message=message,
            step_index=0,
            recoverable=recoverable,
            confidence=confidence,
            details=errors,
        )
        self._last_report = report
        return report
