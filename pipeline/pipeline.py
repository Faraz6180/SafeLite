"""End-to-end execution pipeline for SafeLite."""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from executor.executor import ExecutionEngine
from executor.logger import ExecutionLogger
from executor.models import ExecutionResult
from planner.models import ActionPlan
from planner.planner import PlannerAgent
from safety.models import SafetyResult
from safety.safety_guard import SafetyGuard
from simulator.environment import MetaWorldEnvironment
from self_correction.failure_detector import FailureDetector
from self_correction.models import FailureReport, ReflectionReport, RecoveryDecision
from self_correction.reflection import ReflectionEngine
from self_correction.replanner import Replanner
from self_correction.recovery_strategy import RecoveryStrategyEngine

logger = logging.getLogger(__name__)


class SafeLiteExecutionPipeline:
    """Coordinate planning, safety validation, execution, and self-correction."""

    def __init__(self, log_dir: str | None = None) -> None:
        self.log_dir = Path(log_dir or "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ExecutionLogger(str(self.log_dir))
        self.planner = PlannerAgent()
        self.safety_guard = SafetyGuard()
        self.execution_engine = ExecutionEngine(simulator=None, logger=self.logger)
        self.failure_detector = FailureDetector()
        self.reflection_engine = ReflectionEngine()
        self.replanner = Replanner()
        self.recovery_engine = RecoveryStrategyEngine(replanner=self.replanner)

    def run(self, instruction: str) -> dict[str, Any]:
        """Run the full pipeline for a natural-language instruction."""
        self.logger.info("Starting pipeline for instruction: %s", instruction)

        plan = self.planner.create_plan(instruction)
        self.logger.info("Generated plan with %d actions", len(plan.actions))

        safety_result = self.safety_guard.validate_plan(plan)
        self.logger.info("Safety validation completed")

        execution_result = self.execution_engine.execute_plan(plan, safety_result=safety_result)
        self.logger.info("Execution completed")

        failure_report = self._build_failure_report(execution_result, safety_result)
        reflection_report = None
        recovery_decision = None

        if failure_report is not None:
            reflection_report = self.reflection_engine.analyze_failure(failure_report)
            recovery_decision = self.reflection_engine.recommend_recovery(failure_report)
            self.logger.warning("Self-correction triggered: %s", failure_report.message)

        report = self._build_report(
            instruction=instruction,
            plan=plan,
            safety_result=safety_result,
            execution_result=execution_result,
            failure_report=failure_report,
            reflection_report=reflection_report,
            recovery_decision=recovery_decision,
        )
        self._write_json_report(report)
        self._write_csv_summary(report)
        return report

    def _build_failure_report(self, execution_result: ExecutionResult, safety_result: SafetyResult | None) -> FailureReport | None:
        if execution_result.success:
            return None
        if not execution_result.errors:
            return None
        return self.failure_detector.detect(execution_result, safety_result=safety_result)

    def _build_report(
        self,
        *,
        instruction: str,
        plan: ActionPlan,
        safety_result: SafetyResult,
        execution_result: ExecutionResult,
        failure_report: FailureReport | None,
        reflection_report: ReflectionReport | None,
        recovery_decision: RecoveryDecision | None,
    ) -> dict[str, Any]:
        return {
            "instruction": instruction,
            "generated_plan": plan.model_dump(),
            "safety_report": safety_result.model_dump(),
            "execution_status": self._serialize_execution_result(execution_result),
            "failure_report": failure_report.model_dump() if failure_report else None,
            "reflection_report": reflection_report.model_dump() if reflection_report else None,
            "recovery_decision": recovery_decision.model_dump() if recovery_decision else None,
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": self._build_summary(execution_result, safety_result, failure_report, reflection_report, recovery_decision),
        }

    def _serialize_execution_result(self, execution_result: ExecutionResult) -> dict[str, Any]:
        payload = execution_result.model_dump()
        payload["completed_actions"] = [action.model_dump() for action in execution_result.completed_actions]
        payload["failed_action"] = execution_result.failed_action.model_dump() if execution_result.failed_action else None
        return self._normalize_for_json(payload)

    def _normalize_for_json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._normalize_for_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalize_for_json(item) for item in value]
        if isinstance(value, (datetime,)):
            return value.isoformat()
        if hasattr(value, "model_dump"):
            return self._normalize_for_json(value.model_dump())
        return value

    def _build_summary(
        self,
        execution_result: ExecutionResult,
        safety_result: SafetyResult,
        failure_report: FailureReport | None,
        reflection_report: ReflectionReport | None,
        recovery_decision: RecoveryDecision | None,
    ) -> str:
        if safety_result.approved and execution_result.success:
            return "Execution completed successfully and safely."
        if failure_report is not None and reflection_report is not None:
            return f"Execution failed and self-correction recommended: {recovery_decision.strategy if recovery_decision else 'unknown'}"
        return "Execution completed with warnings or a rejected safety plan."

    def _write_json_report(self, report: dict[str, Any]) -> None:
        path = self.log_dir / "execution_report.json"
        path.write_text(json.dumps(self._normalize_for_json(report), indent=2), encoding="utf-8")

    def _write_csv_summary(self, report: dict[str, Any]) -> None:
        path = self.log_dir / "execution_summary.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["instruction", "status", "safety", "recovery"])
            writer.writerow(
                [
                    report["instruction"],
                    "success" if report["execution_status"]["success"] else "failed",
                    "approved" if report["safety_report"]["approved"] else "rejected",
                    report["recovery_decision"]["strategy"] if report["recovery_decision"] else "none",
                ]
            )
