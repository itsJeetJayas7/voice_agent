"""TTS adapter protocol — defines the interface all TTS transports must implement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass
class PCMFrame:
    """A chunk of PCM audio data."""

    audio: bytes
    sample_rate: int
    channels: int = 1
    samples_per_channel: int = 0


@runtime_checkable
class TTSAdapter(Protocol):
    """Protocol for text-to-speech adapters.

    Implementations must support chunk-at-a-time synthesis and
    return audio frames as an async iterator.
    """

    async def synthesize_stream(
        self,
        text_chunk: str,
        voice: str = "default",
        sample_rate: int = 24000,
    ) -> AsyncIterator[PCMFrame]:
        """Synthesize a text chunk and yield PCM audio frames.

        For streaming providers: forward micro-chunks immediately.
        For non-streaming providers: synthesize per chunk and emit
        split PCM frames (e.g., 20ms packets).

        Args:
            text_chunk: Short speakable text to synthesize.
            voice: Voice identifier.
            sample_rate: Desired output sample rate.
        """
        ...

    async def cancel(self) -> None:
        """Cancel the current synthesis and discard pending audio."""
        ...

    async def close(self) -> None:
        """Close the connection and release resources."""
        ...
