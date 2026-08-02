"""Helpers to package self-correction decisions into a consistent format."""

from __future__ import annotations

from .models import FailureReport, RecoveryDecision, RecoveryStrategy
from .reflection import ReflectionEngine
from .replanner import Replanner


class RecoveryStrategyEngine:
    """Coordinate reflection, decision-making, and replanning."""

    def __init__(self, replanner: Replanner | None = None) -> None:
        self.replanner = replanner or Replanner()
        self.reflection_engine = ReflectionEngine()

    def decide(self, failure: FailureReport) -> tuple[RecoveryDecision, ReflectionReport, RecoveryStrategy]:
        """Produce a recovery decision, reflection, and strategy."""
        decision = self.reflection_engine.recommend_recovery(failure)
        reflection = self.reflection_engine.analyze_failure(failure)
        strategy = self.replanner.build_strategy(failure)
        return decision, reflection, strategy
