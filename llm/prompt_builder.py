"""Reusable prompt construction for LLM-based planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PromptBundle:
    """A complete prompt bundle for a planning request."""

    system_prompt: str
    developer_prompt: str
    user_prompt: str


class PromptBuilder:
    """Create structured prompts for remote providers."""

    def build(self, instruction: str) -> PromptBundle:
        allowed_actions = ["pick", "place", "move", "open", "close", "hold"]
        system_prompt = (
            "You are a safe planning assistant for tabletop robot manipulation. "
            "Return only valid JSON matching the requested schema."
        )
        developer_prompt = (
            "You must produce a JSON object with fields 'goal' and 'actions'. "
            f"Allowed actions: {', '.join(allowed_actions)}. "
            "Use only known objects and locations. Do not include unsafe or malformed steps."
        )
        user_prompt = (
            f"Create a safe manipulation plan for the following instruction: {instruction}. "
            "Return valid JSON with the fields goal and actions."
        )
        return PromptBundle(
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
        )
