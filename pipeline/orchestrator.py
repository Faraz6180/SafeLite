"""High-level orchestrator for the SafeLite execution flow."""

from __future__ import annotations

from .controller import PipelineController


class PipelineOrchestrator:
    """Simple orchestrator wrapper over the pipeline controller."""

    def __init__(self, controller: PipelineController | None = None) -> None:
        self.controller = controller or PipelineController()

    def execute(self, instruction: str) -> dict:
        """Execute the complete pipeline for a natural-language instruction."""
        return self.controller.run_from_input(instruction)
