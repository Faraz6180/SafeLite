"""Factory for constructing LLM provider implementations."""

from __future__ import annotations

from .base import BaseLLMProvider
from .config import LLMConfig
from .groq_provider import GroqProvider
from .huggingface_provider import HuggingFaceProvider
from .mock_provider import MockProvider
from .openrouter_provider import OpenRouterProvider


def create_provider(config: LLMConfig | None = None) -> BaseLLMProvider:
    """Instantiate the provider selected by configuration."""
    cfg = config or LLMConfig()
    provider_name = cfg.provider.lower()

    if provider_name == "mock":
        return MockProvider(cfg)
    if provider_name == "groq":
        if not cfg.api_key:
            return MockProvider(cfg)
        return GroqProvider(cfg)
    if provider_name == "openrouter":
        if not cfg.api_key:
            return MockProvider(cfg)
        return OpenRouterProvider(cfg)
    if provider_name == "huggingface":
        if not cfg.api_key:
            return MockProvider(cfg)
        return HuggingFaceProvider(cfg)

    return MockProvider(cfg)
