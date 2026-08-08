"""
Replanner module for generating new plans after failures.
"""
from __future__ import annotations

from typing import Optional, Any

from self_correction.models import ReflectionReport


class Replanner:
    """
    Replanner that generates a new instruction based on reflection.
    """

    def __init__(self, llm_provider: Optional[Any] = None):
        """
        Initialize the Replanner.

        Args:
            llm_provider: Optional LLM provider for generating new instructions.
        """
        self.llm_provider = llm_provider

    def replan(self, instruction: str, reflection: ReflectionReport) -> str:
        """
        Generate a new instruction based on the reflection.

        Args:
            instruction: The original instruction.
            reflection: The reflection report containing the recommended action.

        Returns:
            A new instruction string.
        """
        # If we have an LLM provider, use it to generate a corrected instruction.
        if self.llm_provider is not None:
            try:
                # Use the LLM to generate a new instruction based on the reflection.
                prompt = f"""
The original instruction was: "{instruction}"

The execution failed. Here is the reflection:
- Summary: {reflection.summary}
- Why it failed: {reflection.why_failed}
- Recommended action: {reflection.recommended_action}

Please provide a corrected instruction that addresses the failure.
"""
                response = self.llm_provider.generate_plan(prompt)
                # Extract the instruction from the response if possible
                if isinstance(response, dict) and 'goal' in response:
                    return response['goal']
                elif isinstance(response, str):
                    return response
                else:
                    # Fallback: append the recommendation to the original instruction
                    return f"{instruction} (Correction: {reflection.recommended_action})"
            except Exception:
                # Fallback if LLM fails
                return f"{instruction} (Correction: {reflection.recommended_action})"
        else:
            # Simple fallback: append the recommended action to the instruction
            return f"{instruction} (Correction: {reflection.recommended_action})"