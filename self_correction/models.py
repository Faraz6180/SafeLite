"""Pydantic models for self-correction reporting and recovery."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from planner.models import Action, ActionPlan


class FailureType(str, Enum):
    """Enumeration of supported failure categories."""

    UNKNOWN_OBJECT = "unknown_object"
    FAILED_GRASP = "failed_grasp"
    OBJECT_NOT_FOUND = "object_not_found"
    INVALID_LOCATION = "invalid_location"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    INVALID_PLAN = "invalid_plan"
    SIMULATOR_ERROR = "simulator_error"
    UNKNOWN_ERROR = "unknown_error"


class FailureReport(BaseModel):
    """Structured report describing a detected execution failure."""

    failure_type: FailureType
    severity: str
    message: str
    step_index: int
    recoverable: bool
    confidence: float
    details: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RecoveryDecision(BaseModel):
    """Decision on how to react to a failure."""

    strategy: str
    recoverable: bool
    reason: str
    confidence: float
    should_terminate: bool = False
    should_replan: bool = False


class ReflectionReport(BaseModel):
    """Explanation and recommendation produced by the reflection engine."""

    summary: str
    why_failed: str
    which_step: str
    recommended_action: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReplannedAction(BaseModel):
    """A single action proposed by the replanner."""

    action_type: str
    target_object: str
    target_location: str
    gripper: str
    reason: str


class RecoveryStrategy(BaseModel):
    """Compact representation of a recovery strategy."""

    name: str
    description: str
    recoverable: bool
    replan_required: bool
