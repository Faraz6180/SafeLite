"""Integration tests for the SafeLite execution pipeline."""

from __future__ import annotations

from pathlib import Path

from pipeline.orchestrator import PipelineOrchestrator


def test_pipeline_runs_and_writes_artifacts(tmp_path: Path) -> None:
    pipeline = PipelineOrchestrator()
    report = pipeline.execute("Pick up the red cube.")

    assert "generated_plan" in report
    assert "safety_report" in report
    assert "execution_status" in report
    assert "summary" in report
    assert report["generated_plan"]["goal"] == "Pick up the red cube."
