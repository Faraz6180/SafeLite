"""Execution history tracking for the execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from planner.models import Action


@dataclass
class ExecutionRecord:
    """A single entry in the execution history."""

    action: Action
    timestamp: datetime
    status: str
    observation: str | None = None
    reward: float | None = None
    error: str | None = None


class ExecutionHistory:
    """Maintain a chronological record of executed actions."""

    def __init__(self) -> None:
        self.records: List[ExecutionRecord] = []

    def add_record(self, record: ExecutionRecord) -> None:
        """Append a record to the history."""
        self.records.append(record)

    def to_list(self) -> List[dict]:
        """Convert the history to a serializable list of dictionaries."""
        return [
            {
                "action": record.action.model_dump(),
                "timestamp": record.timestamp.isoformat(),
                "status": record.status,
                "observation": record.observation,
                "reward": record.reward,
                "error": record.error,
            }
            for record in self.records
        ]
