"""
Data models for safety verification.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class SafetyResult(BaseModel):
    """Result of a safety verification check."""
    valid: bool = Field(..., description="Whether the plan is safe")
    reason: Optional[str] = Field(None, description="Reason for rejection")
    violations: List[str] = Field(default_factory=list, description="List of violations")