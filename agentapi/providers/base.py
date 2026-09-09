"""Provider interface and shared response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class ToolCall:
    """A tool call emitted by the model."""

    id: str
    name: str
    arguments: str


def safe_int_usage(val: Any, default: int = 0) -> int:
    """Safely coerce value to int, returning default if None or invalid."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


@dataclass
class ProviderResponse:
    """Normalized non-stream model response."""

    content: str
    tool_calls: list[ToolCall]
    raw_message: dict[str, Any]
    usage: dict[str, int] | None = None


class BaseProvider(ABC):
    """Abstract provider contract used by Agent."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        pass
