"""SafetyGuard: a validation layer for planner-produced action plans.

The safety guard never executes actions. It evaluates the action plan, records
any violations, and returns a structured safety report.
"""

from __future__ import annotations

import logging
from typing import List

from planner.models import Action, ActionPlan

from .exceptions import SafetyViolationError
from .models import SafetyResult
from .rules import SafetyRules
from .validator import SafetyValidator

logger = logging.getLogger(__name__)


class SafetyGuard:
    """Validate planner-generated action plans before execution.

    The safety guard accepts an ActionPlan and returns a structured validation
    result indicating whether the plan is approved, rejected, or contains
    warnings.
    """

    def __init__(self, rules: SafetyRules | None = None) -> None:
        self.rules = rules or SafetyRules()
        self.validator = SafetyValidator(rules=self.rules)

    def validate_action(self, action: Action) -> None:
        """Validate a single action and raise on violation."""
        self.validator.validate_action(action)

    def validate_plan(self, plan: ActionPlan) -> SafetyResult:
        """Validate an entire ActionPlan and return a safety report."""
        result = SafetyResult(approved=True, violations=[], warnings=[], validated_actions=plan)

        try:
            self.validator.validate_plan(plan)
            logger.info("Plan approved: %s", plan.goal)
        except SafetyViolationError as exc:
            result.approved = False
            result.violations.append(str(exc))
            logger.warning("Plan rejected: %s", exc)

        return result

    def generate_report(self, result: SafetyResult) -> str:
        """Generate a human-readable report from a safety result."""
        if result.approved:
            return "SAFE\nNo violations detected."

        lines = ["UNSAFE"]
        for index, violation in enumerate(result.violations, start=1):
            lines.append(f"Violation {index}: {violation}")
        if result.warnings:
            lines.append("Warnings:")
            for warning in result.warnings:
                lines.append(f"- {warning}")
        return "\n".join(lines)
