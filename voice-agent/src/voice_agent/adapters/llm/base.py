"""LLM adapter protocol — defines the interface for LLM services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass
class LLMToken:
    """A single token from the LLM stream."""

    text: str
    finish_reason: str | None = None


@runtime_checkable
class LLMAdapter(Protocol):
    """Protocol for streaming LLM adapters."""

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[LLMToken]:
        """Stream completion tokens for the given conversation messages.

        Yields LLMToken objects with text content and optional finish_reason.
        Must be cancellation-aware: check for CancelledError frequently.
        """
        ...

    async def cancel(self) -> None:
        """Cancel the current streaming request."""
        ...

    async def close(self) -> None:
        """Close the client and release resources."""
        ...
