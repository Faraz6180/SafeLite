"""
Data models for action plans.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class Action(BaseModel):
    """A single action in a plan."""
    action_type: str = Field(..., description="Type of action: move_to, pick, place, open, close, push")
    target_object: str = Field(..., description="Name of the target object")
    target_location: Optional[str] = Field(None, description="Optional target location")
    gripper: str = Field("hold", description="Gripper state: open, close, hold")
    reason: Optional[str] = Field(None, description="Reason for the action")

    @field_validator('gripper')
    @classmethod
    def validate_gripper(cls, v: str) -> str:
        allowed = {"open", "close", "hold", "closed"}
        if v in allowed:
            # Normalize 'closed' to 'close'
            return "close" if v == "closed" else v
        raise ValueError(f"gripper must be one of: open, close, hold (got {v})")


class ActionPlan(BaseModel):
    """A complete action plan."""
    goal: str = Field(..., description="Overall goal of the plan")
    actions: List[Action] = Field(..., description="List of actions")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw LLM response")