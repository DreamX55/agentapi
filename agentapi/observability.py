"""Observability module for tracking model usage and telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    """Token usage tracking for observability."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def safe_int_usage(val: Any, default: int = 0) -> int:
    """Safely coerce value to int, returning default if None or invalid."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError, OverflowError):
        return default
