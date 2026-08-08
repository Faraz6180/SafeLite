"""
Pipeline Orchestrator - Coordinates Planner, Safety, Executor, and Self-Correction.
"""
import traceback
import time
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
from simulator.environment import MetaWorldEnvironment
from self_correction.models import FailureReport


class SafeLiteExecutionPipeline:
    """
    Orchestrates the full SafeLite pipeline.
    Supports baseline configurations via skip_safety and skip_correction flags.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        simulator: MetaWorldEnvironment,
        skip_safety: bool = False,
        skip_correction: bool = False,
        max_retries: int = 3
    ):
        self.llm_provider = llm_provider
        self.simulator = simulator
        self.skip_safety = skip_safety
        self.skip_correction = skip_correction
        self.max_retries = max_retries

        print("PIPELINE INIT: Creating planner...")
        self.planner = PlannerAgent(provider=llm_provider)

        if not skip_safety:
            print("PIPELINE INIT: Creating safety guard...")
            self.safety_guard = SafetyGuard()
        else:
            self.safety_guard = None

        print("PIPELINE INIT: Creating executor...")
        self.executor = ExecutionEngine(simulator=simulator)

        if not skip_correction:
            print("PIPELINE INIT: Creating correction modules...")
            self.failure_detector = FailureDetector()
            self.reflection_engine = ReflectionEngine()
            self.replanner = Replanner()
            self.recovery_engine = RecoveryStrategyEngine()
        else:
            self.failure_detector = None
            self.reflection_engine = None
            self.replanner = None
            self.recovery_engine = None
        print("PIPELINE INIT: Complete.")

    def run(self, instruction: str) -> ExecutionResult:
        print("\n" + "=" * 60)
        print("PIPELINE: run() called")
        print(f"Instruction: {instruction}")
        print(f"skip_safety: {self.skip_safety}")
        print(f"skip_correction: {self.skip_correction}")
        print("=" * 60)

        attempt = 0
        last_error = None
        current_instruction = instruction
        plan = None
        start_time = time.time()

        while attempt <= self.max_retries:
            print(f"\n--- Attempt {attempt + 1} ---")

            try:
                # 1. Plan
                print("PIPELINE: Calling planner.create_plan()...")
                plan = self.planner.create_plan(current_instruction)
                print(f"PIPELINE: Plan generated with {len(plan.actions)} actions.")
                print(f"PIPELINE: Plan actions: {plan.actions}")

                # 2. Safety verification (optional)
                if self.safety_guard is not None:
                    print("PIPELINE: Calling safety_guard.validate_plan()...")
                    safety_result = self.safety_guard.validate_plan(plan)
                    print(f"PIPELINE: Safety result: valid={safety_result.valid}, reason={safety_result.reason}")
                    if not safety_result.valid:
                        if self.skip_correction:
                            print("PIPELINE: Safety rejected and correction is disabled. Returning failure.")
                            return ExecutionResult(
                                success=False,
                                completed_actions=[],
                                failed_action=None,
                                execution_time=time.time() - start_time,
                                errors=[f"Safety violation: {safety_result.reason}"],
                                logs=["Plan rejected by safety guard."]
                            )
                        else:
                            print("PIPELINE: Safety rejected. Attempting correction...")
                            last_error = f"SafetyGuard: {safety_result.reason}"
                            attempt += 1
                            if attempt > self.max_retries:
                                break
                            # Create FailureReport with required fields
                            failure_report = FailureReport(
                                failure_type="invalid_plan",
                                step_index=0,
                                confidence=1.0,
                                error_code="SAFETY_REJECTION",
                                severity="high",
                                message=safety_result.reason,
                                recoverable=True
                            )
                            reflection = self.reflection_engine.generate_reflection(failure_report)
                            current_instruction = self.replanner.replan(
                                instruction=instruction,   # FIXED: use 'instruction' not 'original_instruction'
                                reflection=reflection
                            )
                            continue
                else:
                    print("PIPELINE: Safety guard is skipped (skip_safety=True).")

                # 3. Execute
                print("PIPELINE: Calling executor.execute_plan()...")
                exec_result = self.executor.execute_plan(plan)
                error_msg = exec_result.errors[0] if exec_result.errors else None
                print(f"PIPELINE: Executor result: success={exec_result.success}, error={error_msg}")

                # 4. If success or correction disabled, return
                if exec_result.success or self.skip_correction:
                    print("PIPELINE: Execution succeeded or correction disabled. Returning result.")
                    return exec_result

                # 5. Correction enabled: detect failure and reflect
                if not self.skip_correction:
                    print("PIPELINE: Execution failed. Attempting correction...")
                    failure_msg = exec_result.errors[0] if exec_result.errors else "Unknown execution failure"
                    # Create FailureReport with required fields
                    failure_report = FailureReport(
                        failure_type="simulator_error",   # valid enum
                        step_index=len(exec_result.completed_actions),
                        confidence=1.0,
                        error_code=failure_msg,
                        severity="high",
                        message=failure_msg,
                        recoverable=False
                    )
                    reflection = self.reflection_engine.generate_reflection(failure_report)
                    current_instruction = self.replanner.replan(
                        instruction=instruction,   # FIXED: use 'instruction'
                        reflection=reflection
                    )
                    attempt += 1
                    last_error = failure_msg
                    continue

            except Exception as e:
                print(f"PIPELINE: EXCEPTION in attempt {attempt+1}: {e}")
                traceback.print_exc()
                return ExecutionResult(
                    success=False,
                    completed_actions=[],
                    failed_action=None,
                    execution_time=time.time() - start_time,
                    errors=[f"Pipeline exception: {str(e)}"],
                    logs=[f"Exception at attempt {attempt+1}"]
                )

        print("PIPELINE: Max retries exceeded. Returning failure.")
        return ExecutionResult(
            success=False,
            completed_actions=[],
            failed_action=None,
            execution_time=time.time() - start_time,
            errors=[f"Max retries exceeded. Last error: {last_error}"],
            logs=["Max retries reached."]
        )