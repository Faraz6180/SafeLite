"""Tests for the SafeLite planner module."""

import pytest

from planner.models import ActionPlan
from planner.parser import ActionPlanParseError, ActionPlanParser
from planner.planner import PlannerAgent
from planner.validator import PlanValidationError, PlanValidator


def test_create_plan_for_valid_instruction() -> None:
    """A valid instruction should produce a structured plan."""
    agent = PlannerAgent()
    plan = agent.create_plan("Pick up the red cube and place it into the blue basket.")

    assert isinstance(plan, ActionPlan)
    assert plan.goal == "Pick up the red cube and place it into the blue basket."
    assert len(plan.actions) == 2


def test_create_plan_rejects_empty_instruction() -> None:
    """An empty instruction should raise a value error."""
    agent = PlannerAgent()

    with pytest.raises(ValueError):
        agent.create_plan("   ")


def test_parser_rejects_malformed_json() -> None:
    """Malformed JSON should raise a meaningful parse error."""
    parser = ActionPlanParser()

    with pytest.raises(ActionPlanParseError):
        parser.parse('{"goal": "x", "actions": [}')


def test_validator_rejects_unknown_actions() -> None:
    """Plans with unknown action types should be rejected."""
    validator = PlanValidator()
    plan = ActionPlan(
        goal="Test plan",
        actions=[
            {
                "action_type": "dance",
                "target_object": "cube",
                "target_location": "box",
                "gripper": "close",
                "reason": "Invalid",
            }
        ],
    )

    with pytest.raises(PlanValidationError):
        validator.validate(plan)


def test_invalid_instruction_result_is_rejected_by_agent() -> None:
    """The planner should reject an invalid instruction structure when parsed."""
    agent = PlannerAgent()
    with pytest.raises(ValueError):
        agent.create_plan("")
