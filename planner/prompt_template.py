"""Reusable prompt templates for the planner agent.

The template instructs a language model to convert a natural-language
instruction into a structured JSON object that matches the planner schema.
"""

from __future__ import annotations


class PlannerPromptTemplate:
    """Reusable prompt template for converting instructions into structured plans."""

    SYSTEM_PROMPT = (
        "You are a planning assistant for robotic manipulation. "
        "Convert the user's instruction into a structured JSON object with: "
        "a goal, a list of actions, and each action containing "
        "action_type, target_object, target_location, gripper, and reason."
    )

    @classmethod
    def build_prompt(cls, instruction: str) -> str:
        """Build a prompt for a language model.

        Parameters
        ----------
        instruction: str
            The natural-language instruction to convert into a plan.

        Returns
        -------
        str
            A fully-formatted prompt for the model.
        """
        return (
            f"{cls.SYSTEM_PROMPT}\n\n"
            f"Instruction: {instruction}\n\n"
            "Return valid JSON matching this structure:\n"
            '{"goal": "...", "actions": [{"action_type": "pick", "target_object": "red cube", "target_location": "blue basket", "gripper": "close", "reason": "..."}]}'
        )
