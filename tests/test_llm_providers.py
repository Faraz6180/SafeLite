"""Tests for the LLM provider abstraction and planner integration."""

from __future__ import annotations

import pytest

from llm.base import LLMProviderError
from llm.config import LLMConfig
from llm.factory import create_provider
from llm.groq_provider import GroqProvider
from llm.mock_provider import MockProvider
from planner.planner import PlannerAgent


class FakeHTTPClient:
    def __init__(self, payload: object | None = None, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def post_json(self, url: str, headers: dict[str, str], payload: dict[str, object]) -> object:
        if self.error is not None:
            raise self.error
        return self.payload


def test_mock_provider_returns_action_plan() -> None:
    provider = MockProvider()
    plan = provider.generate_plan("Pick up the red cube and place it into the blue basket.")

    assert plan.goal == "Pick up the red cube and place it into the blue basket."
    assert len(plan.actions) == 2


def test_groq_provider_parses_json_response() -> None:
    client = FakeHTTPClient(
        {
            "choices": [
                {
                    "message": {
                        "content": '{"goal": "Move the cube", "actions": [{"action_type": "pick", "target_object": "cube", "target_location": "workspace", "gripper": "close", "reason": "Pick"}]}'
                    }
                }
            ]
        }
    )
    provider = GroqProvider(api_key="test-key", http_client=client)
    plan = provider.generate_plan("Move the cube")

    assert plan.goal == "Move the cube"
    assert plan.actions[0].action_type == "pick"


def test_invalid_key_raises_provider_error() -> None:
    client = FakeHTTPClient(
        {"error": {"message": "Invalid API key"}},
        error=None,
    )
    provider = GroqProvider(api_key="invalid-key", http_client=client)

    with pytest.raises(LLMProviderError):
        provider.health_check()


def test_timeout_raises_provider_error() -> None:
    provider = GroqProvider(api_key="test-key", http_client=FakeHTTPClient(error=TimeoutError("timed out")))

    with pytest.raises(LLMProviderError):
        provider.generate_plan("Move the cube")


def test_malformed_response_raises_provider_error() -> None:
    client = FakeHTTPClient({"choices": [{"message": {"content": "not-json"}}]})
    provider = GroqProvider(api_key="test-key", http_client=client)

    with pytest.raises(LLMProviderError):
        provider.generate_plan("Move the cube")


def test_provider_switching_uses_factory() -> None:
    mock_provider = create_provider(LLMConfig(provider="mock"))
    groq_provider = create_provider(LLMConfig(provider="groq", api_key="abc"))

    assert isinstance(mock_provider, MockProvider)
    assert isinstance(groq_provider, GroqProvider)


def test_missing_groq_key_falls_back_to_mock_provider() -> None:
    provider = create_provider(LLMConfig(provider="groq", api_key=None))

    assert isinstance(provider, MockProvider)


def test_planner_agent_uses_mock_provider_by_default() -> None:
    agent = PlannerAgent()
    plan = agent.create_plan("Pick up the red cube and place it into the blue basket.")

    assert plan.goal == "Pick up the red cube and place it into the blue basket."
    assert len(plan.actions) == 2
