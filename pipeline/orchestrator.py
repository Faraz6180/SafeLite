"""
Pipeline Orchestrator - Coordinates Planner, Safety, Executor, and Self-Correction.
"""
from typing import Optional, Dict, Any
from planner.planner import PlannerAgent
from safety.safety_guard import SafetyGuard
from executor.executor import ExecutionEngine
from executor.models import ExecutionResult
from self_correction.reflection import ReflectionEngine
from self_correction.failure_detector import FailureDetector
from self_correction.replanner import Replanner
from self_correction.recovery_strategy import RecoveryStrategyEngine
from llm.base import BaseLLMProvider
from simulator.base import BaseEnvironment


class SafeLiteExecutionPipeline:
    """
    Orchestrates the full SafeLite pipeline.
    Supports baseline configurations via skip_safety and skip_correction flags.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        simulator: BaseEnvironment,
        skip_safety: bool = False,
        skip_correction: bool = False,
        max_retries: int = 3
    ):
        """
        Args:
            llm_provider: LLM provider for planning and reflection.
            simulator: Simulator environment.
            skip_safety: If True, bypass the SafetyGuard (Baseline B0, B1).
            skip_correction: If True, bypass self-correction (Baseline B0, B2).
            max_retries: Maximum number of self-correction attempts.
        """
        self.llm_provider = llm_provider
        self.simulator = simulator
        self.skip_safety = skip_safety
        self.skip_correction = skip_correction
        self.max_retries = max_retries

        # Initialize modules (only if not skipped)
        self.planner = PlannerAgent(llm_provider=llm_provider)

        if not skip_safety:
            self.safety_guard = SafetyGuard()
        else:
            self.safety_guard = None

        self.executor = ExecutionEngine(simulator=simulator)

        if not skip_correction:
            self.failure_detector = FailureDetector()
            self.reflection_engine = ReflectionEngine(llm_provider=llm_provider)
            self.replanner = Replanner(llm_provider=llm_provider)
            self.recovery_engine = RecoveryStrategyEngine()
        else:
            self.failure_detector = None
            self.reflection_engine = None
            self.replanner = None
            self.recovery_engine = None

    def run(self, instruction: str) -> ExecutionResult:
        """
        Execute the pipeline for a given instruction.

        Returns:
            ExecutionResult: Final execution outcome.
        """
        attempt = 0
        last_error = None
        current_instruction = instruction
        plan = None

        while attempt <= self.max_retries:
            # 1. Planning
            plan = self.planner.create_plan(current_instruction)

            # 2. Safety Verification (optional)
            if self.safety_guard is not None:
                safety_result = self.safety_guard.validate_plan(plan)
                if not safety_result.is_valid:
                    # Plan rejected; if correction is disabled, fail immediately
                    if self.skip_correction:
                        return ExecutionResult(
                            success=False,
                            steps_taken=0,
                            error_message=f"Safety violation: {safety_result.reason}",
                            plan=plan
                        )
                    else:
                        last_error = f"SafetyGuard: {safety_result.reason}"
                        attempt += 1
                        if attempt > self.max_retries:
                            break
                        # Use reflection to fix the plan
                        reflection = self.reflection_engine.reflect_on_failure(
                            plan, safety_result.reason
                        )
                        current_instruction = self.replanner.replan(
                            original_instruction=instruction,
                            reflection=reflection
                        )
                        continue

            # 3. Execution
            exec_result = self.executor.execute_plan(plan)

            # 4. If execution succeeded or correction is disabled, return result
            if exec_result.success or self.skip_correction:
                return exec_result

            # 5. Self-Correction (if execution failed and correction is enabled)
            if not self.skip_correction:
                # Detect failure
                failure_report = self.failure_detector.detect(exec_result)
                # Reflect
                reflection = self.reflection_engine.reflect_on_failure(
                    plan, failure_report.error_code
                )
                # Generate new instruction for replanning
                current_instruction = self.replanner.replan(
                    original_instruction=instruction,
                    reflection=reflection
                )
                attempt += 1
                last_error = failure_report.error_code
                continue

        # If we exit the loop, max retries exceeded
        return ExecutionResult(
            success=False,
            steps_taken=0,
            error_message=f"Max retries exceeded. Last error: {last_error}",
            plan=plan
        )