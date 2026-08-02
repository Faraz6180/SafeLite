"""Pydantic models used by the safety guard.

These models describe the approval state of an action plan and the validation
report returned by the safety guard.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from planner.models import ActionPlan


class SafetyResult(BaseModel):
    """The result of safety validation for a planner-produced action plan.

    Attributes
    ----------
    approved: bool
        Whether the plan is safe to proceed.
    violations: List[str]
        Human-readable descriptions of critical violations.
    warnings: List[str]
        Non-blocking warnings that may require attention.
    validated_actions: ActionPlan
        The validated plan object passed through the safety guard.
    """

    approved: bool = Field(default=False)
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validated_actions: ActionPlan
