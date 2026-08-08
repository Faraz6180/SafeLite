"""
Execution engine for SafeLite.
Executes a validated action plan in the simulator.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from executor.models import ExecutionResult
from planner.models import ActionPlan
from simulator.environment import MetaWorldEnvironment


class ExecutionEngine:
    """Execute a plan step by step in the simulator."""

    def __init__(self, simulator: MetaWorldEnvironment):
        self.simulator = simulator

    def execute_plan(self, plan: ActionPlan) -> ExecutionResult:
        """
        Execute the plan in the simulator.
        Returns an ExecutionResult indicating success/failure.
        """
        print("=" * 60)
        print("EXECUTOR: execute_plan() called")
        print(f"Plan has {len(plan.actions)} actions.")
        print(f"Actions: {plan.actions}")
        print("=" * 60)

        if not plan.actions:
            return ExecutionResult(
                success=False,
                completed_actions=[],
                failed_action=None,
                execution_time=0.0,
                errors=["Plan has no actions."],
                logs=["Execution aborted: empty plan."]
            )

        start_time = time.time()
        completed_actions = []
        errors = []
        logs = []

        try:
            for idx, action in enumerate(plan.actions):
                print(f"\n--- Executing action {idx+1}: {action} ---")
                try:
                    result = self.simulator.step(action)
                    completed_actions.append(action)
                    logs.append(f"Action {idx+1} executed successfully.")
                    # If the simulator indicates failure, stop and return failure.
                    if not result.get('success', False):
                        errors.append(f"Action {idx+1} failed in simulator.")
                        return ExecutionResult(
                            success=False,
                            completed_actions=completed_actions,
                            failed_action=action,
                            execution_time=time.time() - start_time,
                            errors=errors,
                            logs=logs
                        )
                except Exception as e:
                    error_msg = f"Exception during action {idx+1}: {e}"
                    print(error_msg)
                    errors.append(error_msg)
                    logs.append(f"Action {idx+1} failed with exception.")
                    return ExecutionResult(
                        success=False,
                        completed_actions=completed_actions,
                        failed_action=action,
                        execution_time=time.time() - start_time,
                        errors=errors,
                        logs=logs
                    )

            # All actions completed
            logs.append("All actions completed successfully.")
            return ExecutionResult(
                success=True,
                completed_actions=completed_actions,
                failed_action=None,
                execution_time=time.time() - start_time,
                errors=errors,
                logs=logs
            )

        except Exception as e:
            error_msg = f"Unhandled exception in execute_plan: {e}"
            print(error_msg)
            errors.append(error_msg)
            return ExecutionResult(
                success=False,
                completed_actions=completed_actions,
                failed_action=None,
                execution_time=time.time() - start_time,
                errors=errors,
                logs=logs
            )