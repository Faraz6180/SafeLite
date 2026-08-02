"""Health check for the Groq provider configuration."""

from __future__ import annotations

import sys

from llm.config import load_config
from llm.factory import create_provider


def main() -> None:
    config = load_config()
    provider = create_provider(config)

    if config.provider != "groq":
        print("✗ Provider is not configured for Groq.")
        sys.exit(1)

    if not config.api_key:
        print("✗ GROQ_API_KEY is missing. Paste your key into .env and rerun.")
        sys.exit(1)

    try:
        provider.generate_plan("Say hello")
    except Exception as exc:  # pragma: no cover - defensive error handling
        print(f"✗ Connected failed: {exc}")
        sys.exit(1)

    print("✓ Connected")


if __name__ == "__main__":
    main()
