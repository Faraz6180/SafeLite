"""Execution engine module for SafeLite.

This package provides an execution layer that processes validated action plans,
communicates with the simulator, tracks execution history, and produces
structured results for later analysis.
"""

from .exceptions import ExecutionError, SimulatorExecutionError, TimeoutError as ExecutionTimeoutError
from .executor import ExecutionEngine
from .history import ExecutionHistory
from .logger import ExecutionLogger
from .models import ExecutionResult, ExecutionStep

__all__ = [
    "ExecutionEngine",
    "ExecutionResult",
    "ExecutionStep",
    "ExecutionHistory",
    "ExecutionLogger",
    "ExecutionError",
    "SimulatorExecutionError",
    "ExecutionTimeoutError",
]
