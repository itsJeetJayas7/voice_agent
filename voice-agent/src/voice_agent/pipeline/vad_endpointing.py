"""VAD + endpointing processor.

Combines voice activity detection with configurable endpointing logic:
- Fast speech-start detection for barge-in responsiveness
- Stable speech-end detection to reduce cutoffs and dead air
- Max utterance cutoff safety

Integrates with the native Pipecat VAD but adds additional endpointing
and state management on top.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Awaitable

from voice_agent.config import Settings
from voice_agent.logging import get_logger

logger = get_logger("vad_endpointing")


class SpeechState(Enum):
    """Current speech detection state."""

    IDLE = auto()
    SPEECH_STARTED = auto()
    SPEECH_ACTIVE = auto()
    SILENCE_AFTER_SPEECH = auto()


@dataclass
class EndpointingConfig:
    """Configurable tunables for VAD + endpointing."""

    vad_confidence: float = 0.6
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.8
    vad_min_volume: float = 0.001
    endpoint_min_silence_ms: int = 300
    endpoint_max_silence_ms: int = 1500
    max_utterance_secs: int = 30

    @classmethod
    def from_settings(cls, settings: Settings) -> EndpointingConfig:
        return cls(
            vad_confidence=settings.vad_confidence,
            vad_start_secs=settings.vad_start_secs,
            vad_stop_secs=settings.vad_stop_secs,
            vad_min_volume=settings.vad_min_volume,
            endpoint_min_silence_ms=settings.endpoint_min_silence_ms,
            endpoint_max_silence_ms=settings.endpoint_max_silence_ms,
            max_utterance_secs=settings.max_utterance_secs,
        )


@dataclass
class EndpointingProcessor:
    """VAD + endpointing state machine.

    Tracks speech start/end transitions and determines when an
    utterance has definitively ended (for STT finalization).

    Callbacks:
        on_speech_start: Called when speech is detected (for barge-in).
        on_speech_end: Called when the utterance is considered complete.
        on_max_utterance: Called when max utterance duration exceeded.
    """

    config: EndpointingConfig
    on_speech_start: Callable[[], Awaitable[None]] | None = None
    on_speech_end: Callable[[], Awaitable[None]] | None = None
    on_max_utterance: Callable[[], Awaitable[None]] | None = None

    # Internal state
    state: SpeechState = field(default=SpeechState.IDLE)
    _speech_start_time: float = 0.0
    _silence_start_time: float = 0.0
    _has_stt_final: bool = False

    def reset(self) -> None:
        """Reset state for a new turn."""
        self.state = SpeechState.IDLE
        self._speech_start_time = 0.0
        self._silence_start_time = 0.0
        self._has_stt_final = False

    async def process_vad_event(self, is_speech: bool, confidence: float = 1.0) -> None:
        """Process a VAD event (speech detected or silence detected).

        Args:
            is_speech: True if voice activity detected.
            confidence: VAD confidence score.
        """
        now = time.monotonic()

        if is_speech and confidence >= self.config.vad_confidence:
            await self._handle_speech(now)
        else:
            await self._handle_silence(now)

    async def notify_stt_final(self) -> None:
        """Notify that STT has produced a final transcript.

        This is used as an additional signal for endpointing —
        if STT says the utterance is done, we can endpoint sooner.
        """
        self._has_stt_final = True
        if self.state == SpeechState.SILENCE_AFTER_SPEECH:
            # STT final + silence = strong endpoint signal
            await self._emit_speech_end()

    async def _handle_speech(self, now: float) -> None:
        if self.state == SpeechState.IDLE:
            self._speech_start_time = now
            self.state = SpeechState.SPEECH_STARTED
            logger.debug("Speech started (tentative)")

        elif self.state == SpeechState.SPEECH_STARTED:
            elapsed = now - self._speech_start_time
            if elapsed >= self.config.vad_start_secs:
                self.state = SpeechState.SPEECH_ACTIVE
                logger.info("Speech active (confirmed)")
                if self.on_speech_start:
                    await self.on_speech_start()

        elif self.state == SpeechState.SILENCE_AFTER_SPEECH:
            # Resume speaking — cancel pending endpoint
            self.state = SpeechState.SPEECH_ACTIVE
            self._silence_start_time = 0.0
            logger.debug("Speech resumed after silence")

        # Check max utterance
        if (
            self.state in (SpeechState.SPEECH_ACTIVE, SpeechState.SPEECH_STARTED)
            and self._speech_start_time
            and (now - self._speech_start_time) > self.config.max_utterance_secs
        ):
            logger.warning("Max utterance duration exceeded")
            if self.on_max_utterance:
                await self.on_max_utterance()
            await self._emit_speech_end()

    async def _handle_silence(self, now: float) -> None:
        if self.state in (SpeechState.SPEECH_ACTIVE, SpeechState.SPEECH_STARTED):
            self._silence_start_time = now
            self.state = SpeechState.SILENCE_AFTER_SPEECH
            logger.debug("Silence after speech detected")

        elif self.state == SpeechState.SILENCE_AFTER_SPEECH:
            silence_ms = (now - self._silence_start_time) * 1000

            # Use min_silence if we have an STT final, max_silence otherwise
            threshold = (
                self.config.endpoint_min_silence_ms
                if self._has_stt_final
                else self.config.endpoint_max_silence_ms
            )

            if silence_ms >= threshold:
                await self._emit_speech_end()

    async def _emit_speech_end(self) -> None:
        """Emit speech end event and reset state."""
        logger.info("Speech ended (endpointed)")
        self.state = SpeechState.IDLE
        self._speech_start_time = 0.0
        self._silence_start_time = 0.0
        self._has_stt_final = False
        if self.on_speech_end:
            await self.on_speech_end()
