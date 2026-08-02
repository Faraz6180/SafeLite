"""Pydantic models for structured robot action plans.

These models define the schema used by the planner agent when converting a
natural-language instruction into a structured action plan.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class Action(BaseModel):
    """A single atomic action in a robot manipulation plan.

    Attributes
    ----------
    action_type: str
        The type of manipulation action to perform.
    target_object: str
        The object that the action should affect.
    target_location: str
        The destination or location associated with the action.
    gripper: str
        The gripper state relative to the action.
    reason: str
        A brief explanation for the action.
    """

    action_type: str = Field(..., min_length=1)
    target_object: str = Field(..., min_length=1)
    target_location: str = Field(..., min_length=1)
    gripper: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, value: str) -> str:
        """Normalize the action type and ensure it is non-empty."""
        return value.strip().lower()

    @field_validator("gripper")
    @classmethod
    def validate_gripper(cls, value: str) -> str:
        """Normalize the gripper command and ensure it is non-empty."""
        normalized = value.strip().lower()
        if normalized not in {"open", "close", "hold"}:
            raise ValueError("gripper must be one of: open, close, hold")
        return normalized


class ActionPlan(BaseModel):
    """A complete structured action plan for a manipulation task.

    Attributes
    ----------
    goal: str
        The high-level manipulation goal.
    actions: List[Action]
        The ordered sequence of actions that fulfill the goal.
    """

    goal: str = Field(..., min_length=1)
    actions: List[Action] = Field(default_factory=list)
