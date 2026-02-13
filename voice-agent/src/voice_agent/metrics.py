"""Latency metrics collection for voice-agent pipeline stages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from voice_agent.logging import get_logger

logger = get_logger("metrics")


@dataclass
class TurnMetrics:
    """Latency metrics for a single conversation turn."""

    turn_id: str = ""
    generation_id: str = ""
    session_id: str = ""

    # Timestamps (monotonic, seconds)
    user_turn_finalized_at: float = 0.0
    llm_request_started_at: float = 0.0
    llm_first_token_at: float = 0.0
    tts_first_chunk_sent_at: float = 0.0
    tts_first_audio_at: float = 0.0
    first_agent_audio_out_at: float = 0.0
    barge_in_speech_start_at: float = 0.0
    barge_in_playback_halted_at: float = 0.0
    endpointer_last_speech_at: float = 0.0
    endpointer_turn_end_at: float = 0.0

    @property
    def llm_ttft_ms(self) -> float:
        """LLM time-to-first-token in milliseconds."""
        if self.llm_request_started_at and self.llm_first_token_at:
            return (self.llm_first_token_at - self.llm_request_started_at) * 1000
        return 0.0

    @property
    def tts_first_audio_latency_ms(self) -> float:
        """TTS first-chunk-sent to first-audio-frame latency."""
        if self.tts_first_chunk_sent_at and self.tts_first_audio_at:
            return (self.tts_first_audio_at - self.tts_first_chunk_sent_at) * 1000
        return 0.0

    @property
    def barge_in_cutoff_ms(self) -> float:
        """Speech start to playback halt latency."""
        if self.barge_in_speech_start_at and self.barge_in_playback_halted_at:
            return (self.barge_in_playback_halted_at - self.barge_in_speech_start_at) * 1000
        return 0.0

    @property
    def e2e_first_audio_ms(self) -> float:
        """User turn finalised to first agent audio out."""
        if self.user_turn_finalized_at and self.first_agent_audio_out_at:
            return (self.first_agent_audio_out_at - self.user_turn_finalized_at) * 1000
        return 0.0

    @property
    def endpointer_latency_ms(self) -> float:
        """Last speech activity to final turn end decision."""
        if self.endpointer_last_speech_at and self.endpointer_turn_end_at:
            return (self.endpointer_turn_end_at - self.endpointer_last_speech_at) * 1000
        return 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "generation_id": self.generation_id,
            "llm_ttft_ms": round(self.llm_ttft_ms, 1),
            "tts_first_audio_latency_ms": round(self.tts_first_audio_latency_ms, 1),
            "barge_in_cutoff_ms": round(self.barge_in_cutoff_ms, 1),
            "e2e_first_audio_ms": round(self.e2e_first_audio_ms, 1),
            "endpointer_latency_ms": round(self.endpointer_latency_ms, 1),
        }

    def emit(self) -> None:
        """Log the turn metrics summary."""
        logger.info(
            "Turn metrics: %s",
            self.summary(),
            extra={"session_id": self.session_id, "turn_id": self.turn_id},
        )


@dataclass
class MetricsCollector:
    """Accumulates per-session turn metrics."""

    session_id: str = ""
    enabled: bool = True
    turns: list[TurnMetrics] = field(default_factory=list)

    def new_turn(self, turn_id: str, generation_id: str) -> TurnMetrics:
        m = TurnMetrics(
            turn_id=turn_id,
            generation_id=generation_id,
            session_id=self.session_id,
        )
        if self.enabled:
            self.turns.append(m)
        return m

    @staticmethod
    def now() -> float:
        """Return monotonic timestamp for latency measurement."""
        return time.monotonic()
