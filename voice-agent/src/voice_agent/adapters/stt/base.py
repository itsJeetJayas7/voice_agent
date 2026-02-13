"""STT adapter protocol — defines the interface all STT transports must implement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator, Protocol, runtime_checkable


class STTEventType(Enum):
    """Types of events emitted by the STT adapter."""

    PARTIAL = auto()       # Interim / partial transcript
    FINAL = auto()         # Final transcript for an utterance
    SPEECH_START = auto()  # VAD-level speech-start from STT service
    SPEECH_END = auto()    # VAD-level speech-end from STT service
    ERROR = auto()         # Error from the STT service


@dataclass
class STTEvent:
    """Event emitted by the STT adapter."""

    type: STTEventType
    text: str = ""
    confidence: float = 0.0
    language: str = ""
    error: str = ""
    is_final: bool = False


@runtime_checkable
class STTAdapter(Protocol):
    """Protocol for speech-to-text adapters.

    Implementations must support incremental audio feeding and
    partial/final transcript callbacks.
    """

    async def connect(self) -> None:
        """Establish connection to the STT service."""
        ...

    async def send_audio(
        self,
        pcm16_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        """Send an audio chunk to the STT service.

        Args:
            pcm16_bytes: PCM16 raw audio bytes.
            sample_rate: Sample rate in Hz.
            channels: Number of audio channels (must be 1 for mono).
        """
        ...

    async def end_utterance(self) -> None:
        """Signal the end of the current utterance to the STT service."""
        ...

    def events(self) -> AsyncIterator[STTEvent]:
        """Async iterator of STT events (partial, final, etc.)."""
        ...

    async def cancel(self) -> None:
        """Cancel the current recognition and discard pending results."""
        ...

    async def close(self) -> None:
        """Close the connection and release resources."""
        ...
