"""Tests for the SafeLite safety guard module."""

from planner.models import Action, ActionPlan

from safety.exceptions import InvalidActionSequenceError, InvalidWorkspaceError, UnknownObjectError
from safety.rules import SafetyRules
from safety.safety_guard import SafetyGuard


def test_valid_plan_is_approved() -> None:
    """A safe plan should be approved."""
    guard = SafetyGuard()
    plan = ActionPlan(
        goal="Move the cube to the basket.",
        actions=[
            Action(action_type="pick", target_object="cube", target_location="workspace", gripper="close", reason="Pick the cube"),
            Action(action_type="place", target_object="cube", target_location="basket", gripper="open", reason="Place the cube"),
        ],
    )

    result = guard.validate_plan(plan)

    assert result.approved is True
    assert result.violations == []


def test_unknown_object_is_rejected() -> None:
    """A plan referencing an unknown object should be rejected."""
    guard = SafetyGuard()
    plan = ActionPlan(
        goal="Pick an unknown object.",
        actions=[
            Action(action_type="pick", target_object="hammer", target_location="workspace", gripper="close", reason="Pick")
        ],
    )

    result = guard.validate_plan(plan)

    assert result.approved is False
    assert any("Unknown object" in violation for violation in result.violations)


def test_invalid_action_is_rejected() -> None:
    """An action with an invalid action type should be rejected."""
    guard = SafetyGuard()
    plan = ActionPlan(
        goal="Perform a disallowed action.",
        actions=[
            Action(action_type="dance", target_object="cube", target_location="workspace", gripper="close", reason="Dance")
        ],
    )

    result = guard.validate_plan(plan)

    assert result.approved is False
    assert any("Unknown action type" in violation for violation in result.violations)


def test_duplicate_action_is_rejected() -> None:
    """Duplicate actions should be rejected."""
    guard = SafetyGuard()
    plan = ActionPlan(
        goal="Repeat the same step.",
        actions=[
            Action(action_type="pick", target_object="cube", target_location="workspace", gripper="close", reason="Pick"),
            Action(action_type="pick", target_object="cube", target_location="basket", gripper="open", reason="Pick again"),
        ],
    )

    result = guard.validate_plan(plan)

    assert result.approved is False
    assert any("Duplicate action" in violation for violation in result.violations)


def test_out_of_bounds_location_is_rejected() -> None:
    """A plan with an unknown workspace location should be rejected."""
    guard = SafetyGuard()
    plan = ActionPlan(
        goal="Place the object somewhere invalid.",
        actions=[
            Action(action_type="place", target_object="cube", target_location="moon", gripper="open", reason="Place")
        ],
    )

    result = guard.validate_plan(plan)

    assert result.approved is False
    assert any("Unknown location" in violation for violation in result.violations)


def test_empty_plan_is_rejected() -> None:
    """An empty plan should be rejected."""
    guard = SafetyGuard()
    plan = ActionPlan.model_construct(goal="", actions=[])

    result = guard.validate_plan(plan)

    assert result.approved is False
    assert any("Empty plan" in violation for violation in result.violations)
