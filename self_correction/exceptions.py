"""Custom exceptions for the self-correction module."""

from __future__ import annotations


class SelfCorrectionError(RuntimeError):
    """Raised when self-correction cannot process a failure safely."""
