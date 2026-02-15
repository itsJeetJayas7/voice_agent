"""Adaptive TTS chunker — splits streaming LLM tokens into speakable chunks.

Flush rules:
1. Punctuation boundary reached and min_chars met
2. OR max_chars reached
3. OR max_wait_ms elapsed since first unflushed token

Prioritises natural phrase boundaries (commas, periods, questions, etc.)
while keeping latency low by enforcing a time deadline.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from pipecat.utils.text.base_text_aggregator import (
    Aggregation,
    AggregationType,
    BaseTextAggregator,
)

from voice_agent.config import Settings
from voice_agent.logging import get_logger

logger = get_logger("tts_chunker")

# Characters that form natural speech boundaries
PUNCTUATION_BOUNDARIES = frozenset(".!?;:,")
# Stronger boundaries — always flush here if min_chars met
STRONG_BOUNDARIES = frozenset(".!?")


@dataclass
class ChunkerConfig:
    """Configuration for adaptive TTS chunking."""

    min_chars: int = 24
    max_chars: int = 120
    max_wait_ms: int = 350
    queue_maxsize: int = 2

    @classmethod
    def from_settings(cls, settings: Settings) -> ChunkerConfig:
        return cls(
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
            max_wait_ms=settings.chunk_max_wait_ms,
        )


class AdaptiveTextAggregator(BaseTextAggregator):
    """Low-latency text aggregator for streaming LLM -> TTS.

    Flush rules match ``TTSChunker``:
    1. `max_chars` reached
    2. punctuation boundary reached and `min_chars` met
    3. `max_wait_ms` deadline reached
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self._buffer: str = ""
        self._first_token_time: float = 0.0

    @property
    def text(self) -> Aggregation:
        return Aggregation(text=self._buffer.strip(" "), type=AggregationType.SENTENCE)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        if not text:
            return

        self._buffer += text
        if not self._first_token_time and self._buffer.strip():
            self._first_token_time = time.monotonic()

        while True:
            chunk = self._next_chunk()
            if not chunk:
                break
            yield Aggregation(text=chunk, type=AggregationType.SENTENCE)

    async def flush(self) -> Optional[Aggregation]:
        if self._buffer.strip():
            result = self._buffer.strip()
            await self.reset()
            return Aggregation(text=result, type=AggregationType.SENTENCE)
        return None

    async def handle_interruption(self):
        await self.reset()

    async def reset(self):
        self._buffer = ""
        self._first_token_time = 0.0

    def _next_chunk(self) -> str | None:
        text = self._buffer
        if not text:
            return None

        if len(text) >= self.config.max_chars:
            split_at = self._find_break_index(text, self.config.max_chars)
            return self._drain_prefix(split_at)

        if len(text) >= self.config.min_chars:
            punctuation_idx = self._find_punctuation_boundary(text)
            if punctuation_idx is not None:
                return self._drain_prefix(punctuation_idx + 1)

        if (
            self._first_token_time
            and text.strip()
            and (time.monotonic() - self._first_token_time) * 1000 >= self.config.max_wait_ms
        ):
            return self._drain_prefix(len(text))

        return None

    def _find_break_index(self, text: str, max_len: int) -> int:
        limit = min(max_len, len(text))
        for i in range(limit - 1, -1, -1):
            if text[i] in PUNCTUATION_BOUNDARIES:
                return i + 1
        return limit

    def _find_punctuation_boundary(self, text: str) -> int | None:
        start = max(self.config.min_chars - 1, 0)
        for i in range(len(text) - 1, start - 1, -1):
            if text[i] in PUNCTUATION_BOUNDARIES:
                return i
        return None

    def _drain_prefix(self, size: int) -> str | None:
        chunk = self._buffer[:size].strip()
        self._buffer = self._buffer[size:].lstrip()
        self._first_token_time = time.monotonic() if self._buffer.strip() else 0.0
        return chunk or None


class TTSChunker:
    """Accumulates LLM tokens and flushes speakable chunks.

    Thread-safe for single-producer (LLM stream) use.
    Output queue is bounded (maxsize 2) for backpressure.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self._buffer: str = ""
        self._first_token_time: float = 0.0
        self._output_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.config.queue_maxsize)
        self._done = False
        self._cancelled = False

    def reset(self) -> None:
        """Reset chunker state for a new turn."""
        self._buffer = ""
        self._first_token_time = 0.0
        self._done = False
        self._cancelled = False
        # Drain output queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def add_token(self, token: str) -> None:
        """Add a token from the LLM stream.

        May trigger a flush if boundary conditions are met.
        """
        if self._cancelled:
            return

        self._buffer += token

        if not self._first_token_time:
            self._first_token_time = time.monotonic()

        # Check flush conditions
        await self._maybe_flush()

    async def flush_remaining(self) -> None:
        """Flush any remaining text (called on LLM stream end)."""
        if self._buffer.strip():
            await self._emit(self._buffer.strip())
            self._buffer = ""
        self._done = True
        self._first_token_time = 0.0
        # Signal end-of-stream
        await self._output_queue.put("")

    async def _maybe_flush(self) -> None:
        """Check if we should flush the current buffer."""
        text = self._buffer

        # Rule 1: Max chars reached — force flush
        if len(text) >= self.config.max_chars:
            # Try to break at a natural boundary
            flush_text = self._find_break_point(text, self.config.max_chars)
            if flush_text:
                await self._emit(flush_text)
                return

        # Rule 2: Punctuation boundary + min_chars met
        if len(text) >= self.config.min_chars:
            # Check for punctuation at or near the end
            for i in range(len(text) - 1, max(self.config.min_chars - 2, 0) - 1, -1):
                if text[i] in PUNCTUATION_BOUNDARIES:
                    flush_text = text[: i + 1].strip()
                    if flush_text:
                        self._buffer = text[i + 1 :]
                        self._first_token_time = time.monotonic() if self._buffer.strip() else 0.0
                        await self._output_queue.put(flush_text)
                        return
                    break

        # Rule 3: Max wait elapsed
        if (
            self._first_token_time
            and len(text.strip()) >= 1
            and (time.monotonic() - self._first_token_time) * 1000 >= self.config.max_wait_ms
        ):
            flush_text = text.strip()
            if flush_text:
                await self._emit(flush_text)

    def _find_break_point(self, text: str, max_len: int) -> str | None:
        """Find the best break point in text up to max_len characters."""
        # Look for the last punctuation boundary within max_len
        for i in range(min(max_len, len(text)) - 1, -1, -1):
            if text[i] in PUNCTUATION_BOUNDARIES:
                result = text[: i + 1].strip()
                self._buffer = text[i + 1 :]
                self._first_token_time = time.monotonic() if self._buffer.strip() else 0.0
                return result

        # No punctuation found — break at max_len
        result = text[:max_len].strip()
        self._buffer = text[max_len:]
        self._first_token_time = time.monotonic() if self._buffer.strip() else 0.0
        return result

    async def _emit(self, text: str) -> None:
        """Emit a chunk to the output queue."""
        self._buffer = self._buffer.lstrip()
        self._first_token_time = time.monotonic() if self._buffer.strip() else 0.0
        if text:
            await self._output_queue.put(text)
            logger.debug("Chunk emitted (%d chars): %.40s...", len(text), text)

    async def chunks(self) -> asyncio.Queue[str]:
        """Return the output queue for consuming speakable chunks.

        Empty string signals end of stream.
        """
        return self._output_queue

    def cancel(self) -> None:
        """Cancel chunking — discard accumulated text."""
        self._cancelled = True
        self._buffer = ""
        self._first_token_time = 0.0
        # Drain queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
