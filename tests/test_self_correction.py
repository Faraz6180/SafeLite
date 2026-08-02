"""Tests for the self-correction module."""

from executor.models import ExecutionResult
from planner.models import Action, ActionPlan

from self_correction.failure_detector import FailureDetector
from self_correction.models import FailureReport, FailureType
from self_correction.reflection import ReflectionEngine
from self_correction.replanner import Replanner
from self_correction.recovery_strategy import RecoveryStrategyEngine


def make_execution_result(errors: list[str]) -> ExecutionResult:
    return ExecutionResult(success=False, completed_actions=[], errors=errors)


def test_timeout_failure_is_recoverable() -> None:
    detector = FailureDetector()
    failure = detector.detect(make_execution_result(["Timeout: action exceeded deadline"]))

    assert failure.failure_type == FailureType.TIMEOUT
    assert failure.recoverable is True

    reflection = ReflectionEngine().analyze_failure(failure)
    assert "Retry" in reflection.recommended_action

    decision = ReflectionEngine().recommend_recovery(failure)
    assert decision.recoverable is True
    assert decision.strategy == "retry_current_action"


def test_collision_failure_terminates_safely() -> None:
    detector = FailureDetector()
    failure = detector.detect(
        make_execution_result(["Unexpected collision during placement"]),
        simulator_feedback="collision detected",
    )

    assert failure.failure_type == FailureType.COLLISION
    assert failure.recoverable is False

    decision = ReflectionEngine().recommend_recovery(failure)
    assert decision.should_terminate is True
    assert decision.recoverable is False


def test_invalid_plan_replanning_is_requested() -> None:
    detector = FailureDetector()
    failure = detector.detect(make_execution_result(["invalid plan"]), simulator_feedback=None)
    failure.step_index = 1

    strategy_engine = RecoveryStrategyEngine(replanner=Replanner())
    decision, reflection, strategy = strategy_engine.decide(failure)

    assert decision.should_replan is True
    assert strategy.replan_required is True
    assert "replan" in strategy.name.lower()


def test_replanner_returns_safe_action_for_unknown_object() -> None:
    failure = FailureReport(
        failure_type=FailureType.UNKNOWN_OBJECT,
        severity="high",
        message="unknown object",
        step_index=0,
        recoverable=False,
        confidence=0.95,
        details=["unknown object"],
    )

    plan = ActionPlan(
        goal="Move the cube",
        actions=[
            Action(action_type="pick", target_object="cube", target_location="workspace", gripper="close", reason="Pick"),
        ],
    )

    replanner = Replanner()
    actions = replanner.replan(failure, current_plan=plan)

    assert len(actions) == 1
    assert actions[0].action_type == "replace_target"
    assert actions[0].target_object == "object"
