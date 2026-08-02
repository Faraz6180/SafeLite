"""Controller for the SafeLite execution pipeline."""

from __future__ import annotations

import logging
from typing import Any

from .pipeline import SafeLiteExecutionPipeline

logger = logging.getLogger(__name__)


class PipelineController:
    """Coordinate operator input and pipeline execution."""

    def __init__(self, pipeline: SafeLiteExecutionPipeline | None = None) -> None:
        self.pipeline = pipeline or SafeLiteExecutionPipeline()

    def run_from_input(self, instruction: str) -> dict[str, Any]:
        """Run the pipeline for a user-provided instruction."""
        try:
            return self.pipeline.run(instruction)
        except Exception as exc:  # pragma: no cover - defensive error handling
            logger.exception("Pipeline execution failed.")
            return {
                "instruction": instruction,
                "error": str(exc),
                "summary": "Pipeline execution failed due to an unexpected exception.",
            }
