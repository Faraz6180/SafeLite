"""Execution logging utilities.

The logging module writes detailed execution information to the console and to
files under the project logs directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List


class ExecutionLogger:
    """Create and manage execution logs for the engine."""

    def __init__(self, log_dir: str | None = None) -> None:
        self.log_dir = Path(log_dir or "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("safelite.executor")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(self.log_dir / "execution.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def info(self, message: str, *args: object) -> None:
        """Log an informational message."""
        self.logger.info(message, *args)

    def warning(self, message: str, *args: object) -> None:
        """Log a warning message."""
        self.logger.warning(message, *args)

    def error(self, message: str, *args: object) -> None:
        """Log an error message."""
        self.logger.error(message, *args)

    def export_logs(self) -> List[str]:
        """Return the log contents as strings."""
        log_path = self.log_dir / "execution.log"
        if not log_path.exists():
            return []
        return log_path.read_text(encoding="utf-8").splitlines()
