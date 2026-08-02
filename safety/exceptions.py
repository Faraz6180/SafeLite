"""Custom exceptions for the safety guard."""

from __future__ import annotations


class SafetyViolationError(ValueError):
    """Raised when a safety rule is violated."""


class InvalidWorkspaceError(SafetyViolationError):
    """Raised when an action targets a location outside the allowed workspace."""


class UnknownObjectError(SafetyViolationError):
    """Raised when an action references an unknown object."""


class InvalidActionSequenceError(SafetyViolationError):
    """Raised when an action sequence violates ordering constraints."""
