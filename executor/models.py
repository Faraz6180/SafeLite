"""Pydantic models for execution results and execution steps."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from planner.models import Action


class ExecutionStep(BaseModel):
    """A single step in an executed action plan.

    Attributes
    ----------
    action: Action
        The action that was attempted.
    status: str
        The status of the action, such as completed or failed.
    start_time: datetime
        When execution of the action started.
    end_time: datetime
        When execution of the action ended.
    message: str
        A human-readable message describing the outcome.
    """

    action: Action
    status: str = Field(default="pending")
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    message: str = Field(default="")


class ExecutionResult(BaseModel):
    """The overall outcome of executing an action plan.

    Attributes
    ----------
    success: bool
        Whether the overall plan execution succeeded.
    completed_actions: List[Action]
        The actions that completed successfully.
    failed_action: Optional[Action]
        The action that caused execution to fail, if any.
    execution_time: float
        The total wall-clock execution time in seconds.
    errors: List[str]
        Any errors encountered during execution.
    logs: List[str]
        The detailed logs emitted during execution.
    """

    success: bool = Field(default=False)
    completed_actions: List[Action] = Field(default_factory=list)
    failed_action: Optional[Action] = None
    execution_time: float = Field(default=0.0)
    errors: List[str] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
