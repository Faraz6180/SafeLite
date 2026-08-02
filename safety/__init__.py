"""Safety guard module for SafeLite.

This package validates action plans produced by the planner before any
execution or simulator interaction occurs.
"""

from .exceptions import (
    InvalidActionSequenceError,
    InvalidWorkspaceError,
    SafetyViolationError,
    UnknownObjectError,
)
from .models import SafetyResult
from .rules import SafetyRules
from .safety_guard import SafetyGuard
from .validator import SafetyValidator

__all__ = [
    "SafetyGuard",
    "SafetyResult",
    "SafetyRules",
    "SafetyValidator",
    "SafetyViolationError",
    "InvalidWorkspaceError",
    "UnknownObjectError",
    "InvalidActionSequenceError",
]
