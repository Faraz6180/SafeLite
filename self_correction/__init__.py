"""Self-correction module for SafeLite.

This package analyzes execution failures, classifies them, and proposes
recovery actions or safe termination without directly executing the plan.
"""

from .exceptions import SelfCorrectionError
from .failure_detector import FailureDetector
from .models import FailureReport, RecoveryDecision, ReflectionReport, RecoveryStrategy, ReplannedAction
from .reflection import ReflectionEngine
from .replanner import Replanner

__all__ = [
    "FailureDetector",
    "ReflectionEngine",
    "Replanner",
    "FailureReport",
    "RecoveryDecision",
    "ReflectionReport",
    "RecoveryStrategy",
    "ReplannedAction",
    "SelfCorrectionError",
]
