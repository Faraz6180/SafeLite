"""Abstract base class and errors for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from planner.models import ActionPlan


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider fails to produce a valid plan."""


class BaseLLMProvider(ABC):
    """Abstract interface for providers that generate action plans."""

    def __init__(self, config: Any | None = None) -> None:
        self.config = config

    @abstractmethod
    def generate_plan(self, instruction: str) -> ActionPlan:
        """Generate a structured action plan from natural language."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return whether the provider is healthy and reachable."""

    @abstractmethod
    def validate_response(self, payload: Any) -> ActionPlan:
        """Validate and normalize a provider-specific response into an ActionPlan."""
