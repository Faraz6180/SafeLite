"""Custom exceptions for the execution engine."""

from __future__ import annotations


class ExecutionError(RuntimeError):
    """Raised when execution fails for a known reason."""


class SimulatorExecutionError(ExecutionError):
    """Raised when the simulator reports an execution failure."""


class TimeoutError(ExecutionError):
    """Raised when an action exceeds the allowed execution time."""
