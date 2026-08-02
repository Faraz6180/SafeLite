"""Execution engine for validated action plans.

The execution engine receives an ActionPlan from the safety guard, executes
its actions sequentially against a simulator-like interface, records each step,
and returns a structured execution result.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, List, Optional

from planner.models import Action, ActionPlan
from safety.models import SafetyResult

from .exceptions import ExecutionError, SimulatorExecutionError, TimeoutError as ExecutionTimeoutError
from .history import ExecutionHistory, ExecutionRecord
from .logger import ExecutionLogger
from .models import ExecutionResult, ExecutionStep


class ExecutionEngine:
    """Execute a validated action plan and record its outcome."""

    def __init__(self, simulator: Any | None = None, logger: ExecutionLogger | None = None) -> None:
        self.simulator = simulator
        self.logger = logger or ExecutionLogger()
        self.history = ExecutionHistory()
        self._stop_requested = False

    def execute_plan(self, plan: ActionPlan, safety_result: SafetyResult | None = None) -> ExecutionResult:
        """Execute an action plan sequentially and return the outcome."""
        self._stop_requested = False
        start_time = time.time()
        result = ExecutionResult(success=False, completed_actions=[], errors=[], logs=[])

        self.logger.info("Start execution")
        result.logs.append("Start execution")

        if not plan.actions:
            message = "Empty plan: no actions to execute."
            self.logger.warning(message)
            result.errors.append(message)
            result.logs.append(message)
            result.execution_time = time.time() - start_time
            return result

        if safety_result is not None and not safety_result.approved:
            message = "Plan was rejected by the safety guard; execution aborted."
            self.logger.warning(message)
            result.errors.append(message)
            result.logs.append(message)
            result.execution_time = time.time() - start_time
            return result

        for index, action in enumerate(plan.actions, start=1):
            if self._stop_requested:
                self.logger.warning("Execution terminated by stop request.")
                result.logs.append("Execution terminated by stop request.")
                break

            step = self.execute_action(action, step_index=index)
            result.logs.append(step.message)
            self.history.add_record(
                ExecutionRecord(
                    action=action,
                    timestamp=datetime.now(UTC),
                    status=step.status,
                    observation=step.message,
                    reward=None,
                    error=None if step.status == "completed" else step.message,
                )
            )

            if step.status == "completed":
                result.completed_actions.append(action)
            else:
                result.failed_action = action
                result.errors.append(step.message)
                break

        result.success = len(result.completed_actions) == len(plan.actions)
        result.execution_time = time.time() - start_time
        result.logs.extend(self.export_execution_log())
        return result

    def execute_action(self, action: Action, step_index: int) -> ExecutionStep:
        """Execute a single action and return its status."""
        step = ExecutionStep(action=action, status="running", start_time=datetime.now(UTC), message="")
        self.logger.info(f"Execute Action {step_index}: {action.action_type} -> {action.target_object}")

        if self.simulator is None:
            step.status = "failed"
            step.end_time = datetime.now(UTC)
            step.message = "No simulator available."
            self.logger.error(step.message)
            return step

        try:
            if not hasattr(self.simulator, "step"):
                raise SimulatorExecutionError("Simulator does not expose a step method.")

            self.simulator.step(action.model_dump())
            step.status = "completed"
            step.end_time = datetime.now(UTC)
            step.message = f"Action completed: {action.action_type}"
            self.logger.info(step.message)
        except TimeoutError as exc:
            step.status = "failed"
            step.end_time = datetime.now(UTC)
            step.message = f"Timeout: {exc}"
            self.logger.error(step.message)
        except SimulatorExecutionError as exc:
            step.status = "failed"
            step.end_time = datetime.now(UTC)
            step.message = f"Simulator failure: {exc}"
            self.logger.error(step.message)
        except Exception as exc:  # pragma: no cover - defensive error handling
            step.status = "failed"
            step.end_time = datetime.now(UTC)
            step.message = f"Unexpected exception: {exc}"
            self.logger.error(step.message)

        return step

    def record_step(self, step: ExecutionStep) -> None:
        """Record a completed or failed execution step."""
        self.logger.info(f"Recording step: {step.status} - {step.message}")

    def stop_execution(self) -> None:
        """Request that execution stop gracefully."""
        self._stop_requested = True
        self.logger.warning("Execution stopped.")

    def export_execution_log(self) -> List[str]:
        """Export the execution log as a list of strings."""
        return self.logger.export_logs()
