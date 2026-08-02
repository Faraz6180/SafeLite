"""Configuration helpers for selecting and configuring an LLM provider."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(slots=True)
class LLMConfig:
    """Runtime configuration for the active LLM provider."""

    provider: str = "mock"
    api_key: str | None = None
    model_name: str | None = None
    temperature: float = 0.1
    max_tokens: int = 512
    timeout: int = 30


def load_config(env_path: str | None = None) -> LLMConfig:
    """Load LLM configuration from the environment or a .env file."""
    path = Path(env_path or ".env")
    if path.exists():
        load_dotenv(path, override=False)
    else:
        load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider not in {"mock", "groq", "openrouter", "huggingface"}:
        provider = "mock"

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("HF_API_KEY")
    if provider == "groq" and not api_key:
        api_key = None
    if provider == "openrouter" and not api_key:
        api_key = None
    if provider == "huggingface" and not api_key:
        api_key = None

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model_name=os.getenv("MODEL_NAME"),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("MAX_TOKENS", "512")),
        timeout=int(os.getenv("TIMEOUT", "30")),
    )
