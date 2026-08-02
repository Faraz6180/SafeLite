"""Configurable safety rules for action-plan validation.

The rules are intentionally reusable and parameterized so the safety guard can
be adapted without hardcoding every constraint in the validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass(frozen=True)
class SafetyRules:
    """A configurable collection of safety rules.

    Attributes
    ----------
    workspace_limits: Dict[str, tuple[float, float]]
        Allowed coordinate bounds for each workspace dimension.
    max_reach: float
        Maximum allowed reach distance for a single action.
    allowed_actions: Set[str]
        Allowed action types.
    forbidden_objects: Set[str]
        Objects that are forbidden by policy.
    allowed_objects: Set[str]
        Objects that are permitted in the workspace.
    allowed_locations: Set[str]
        Allowed target locations.
    allowed_grippers: Set[str]
        Allowed gripper states.
    """

    workspace_limits: Dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (0.0, 1.0),
        }
    )
    max_reach: float = 1.0
    allowed_actions: Set[str] = field(
        default_factory=lambda: {"pick", "place", "move", "open", "close", "hold"}
    )
    forbidden_objects: Set[str] = field(default_factory=set)
    allowed_objects: Set[str] = field(default_factory=lambda: {"cube", "box", "basket", "object"})
    allowed_locations: Set[str] = field(default_factory=lambda: {"workspace", "table", "basket", "box"})
    allowed_grippers: Set[str] = field(default_factory=lambda: {"open", "close", "hold"})
