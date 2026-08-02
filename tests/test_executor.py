"""Tests for the SafeLite execution engine."""

from planner.models import Action, ActionPlan
from safety.models import SafetyResult

from executor.exceptions import SimulatorExecutionError
from executor.executor import ExecutionEngine
from executor.models import ExecutionStep


class StubSimulator:
    """Simple simulator stub for execution tests."""

    def __init__(self, fail_on_step: int | None = None) -> None:
        self.fail_on_step = fail_on_step
        self.calls: list[int] = []

    def step(self, action: dict) -> None:
        self.calls.append(1)
        if self.fail_on_step is not None and len(self.calls) >= self.fail_on_step:
            raise SimulatorExecutionError("simulator failure")


def test_successful_execution() -> None:
    """A simple valid plan should execute successfully."""
    engine = ExecutionEngine(simulator=StubSimulator())
    plan = ActionPlan(
        goal="Move the cube",
        actions=[
            Action(action_type="pick", target_object="cube", target_location="workspace", gripper="close", reason="Pick"),
        ],
    )

    result = engine.execute_plan(plan)

    assert result.success is True
    assert len(result.completed_actions) == 1


def test_simulator_failure() -> None:
    """A simulator step failure should be recorded as an execution failure."""
    engine = ExecutionEngine(simulator=StubSimulator(fail_on_step=1))
    plan = ActionPlan(
        goal="Move the cube",
        actions=[
            Action(action_type="pick", target_object="cube", target_location="workspace", gripper="close", reason="Pick"),
        ],
    )

    result = engine.execute_plan(plan)

    assert result.success is False
    assert result.failed_action is not None


def test_empty_plan() -> None:
    """An empty plan should fail gracefully."""
    engine = ExecutionEngine(simulator=StubSimulator())
    plan = ActionPlan(goal="Empty", actions=[])

    result = engine.execute_plan(plan)

    assert result.success is False
    assert result.errors


def test_partial_execution() -> None:
    """Execution should stop once a failure is encountered."""
    engine = ExecutionEngine(simulator=StubSimulator(fail_on_step=2))
    plan = ActionPlan(
        goal="Two-step execution",
        actions=[
            Action(action_type="pick", target_object="cube", target_location="workspace", gripper="close", reason="Pick"),
            Action(action_type="place", target_object="cube", target_location="basket", gripper="open", reason="Place"),
        ],
    )

    result = engine.execute_plan(plan)

    assert result.success is False
    assert result.failed_action is not None


def test_unexpected_exception() -> None:
    """An unexpected simulator exception should be recorded as a failure."""

    class ExplodingSimulator:
        def step(self, action: dict) -> None:
            raise RuntimeError("boom")

    engine = ExecutionEngine(simulator=ExplodingSimulator())
    plan = ActionPlan(
        goal="Unexpected failure",
        actions=[
            Action(action_type="move", target_object="cube", target_location="workspace", gripper="hold", reason="Move"),
        ],
    )

    result = engine.execute_plan(plan)

    assert result.success is False
    assert result.failed_action is not None
