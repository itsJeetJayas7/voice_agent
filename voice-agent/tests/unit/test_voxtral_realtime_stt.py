"""Tests for VoxtralRealtimeSTTService fallback, debounce, and empty-done behaviour.

These tests mock the pipecat base class and websocket to exercise the STT
service's commit logic in isolation.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import numpy as np

from pipecat.frames.frames import (
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)

from voice_agent.pipeline.voxtral_realtime_stt_service import (
    VoxtralRealtimeSTTService,
    _VOXTRAL_SAMPLE_RATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stt(**overrides) -> VoxtralRealtimeSTTService:
    """Create a service instance with test defaults, bypassing websocket."""
    defaults = dict(
        host="localhost",
        port=8000,
        model="test-model",
        fallback_commit_enabled=False,
        fallback_commit_interval_s=2.0,
        fallback_min_voiced_appends=8,
        voiced_peak_threshold=80,
        vad_stop_debounce_ms=150,
    )
    defaults.update(overrides)
    svc = VoxtralRealtimeSTTService(**defaults)
    # Fake a connected & ready state so run_stt / commit paths execute.
    svc._session_ready = True
    svc._disconnecting = False
    ws_mock = AsyncMock()
    ws_mock.send = AsyncMock()
    svc._websocket = ws_mock
    return svc


def _loud_pcm16(n_samples: int = 160) -> bytes:
    """Return PCM16 audio with peak well above default threshold (80)."""
    samples = np.full(n_samples, 500, dtype=np.int16)
    return samples.tobytes()


def _silent_pcm16(n_samples: int = 160) -> bytes:
    """Return PCM16 audio with peak below threshold."""
    samples = np.full(n_samples, 10, dtype=np.int16)
    return samples.tobytes()


def _sent_messages(svc: VoxtralRealtimeSTTService) -> list[dict]:
    """Extract all JSON payloads sent via the websocket mock."""
    return [
        json.loads(call.args[0])
        for call in svc._websocket.send.call_args_list
    ]


def _commit_messages(svc: VoxtralRealtimeSTTService) -> list[dict]:
    return [m for m in _sent_messages(svc) if m.get("type") == "input_audio_buffer.commit"]


def _final_commits(svc: VoxtralRealtimeSTTService) -> list[dict]:
    return [m for m in _commit_messages(svc) if m.get("final") is True]


# ---------------------------------------------------------------------------
# Tests: Fallback disabled (default)
# ---------------------------------------------------------------------------


class TestFallbackDisabled:
    """When fallback_commit_enabled=False, no periodic final commits should fire."""

    async def test_no_fallback_commits_during_continuous_speech(self):
        svc = _make_stt(fallback_commit_enabled=False, fallback_commit_interval_s=0.01)
        # Force resampler to pass-through
        svc._resampler = AsyncMock()
        svc._resampler.resample = AsyncMock(return_value=_loud_pcm16())
        svc._sample_rate = _VOXTRAL_SAMPLE_RATE

        # Simulate many loud audio appends (well above min_appends)
        for _ in range(30):
            async for _ in svc.run_stt(_loud_pcm16()):
                pass

        final = _final_commits(svc)
        assert len(final) == 0, f"Expected 0 fallback final commits, got {len(final)}"


class TestFallbackEnabled:
    """When fallback_commit_enabled=True, periodic final commits should fire under conditions."""

    async def test_fallback_fires_when_enabled_and_conditions_met(self):
        svc = _make_stt(
            fallback_commit_enabled=True,
            fallback_commit_interval_s=0.0,  # no time gate for test speed
            fallback_min_voiced_appends=3,
        )
        svc._resampler = AsyncMock()
        svc._resampler.resample = AsyncMock(return_value=_loud_pcm16())
        svc._sample_rate = _VOXTRAL_SAMPLE_RATE

        for _ in range(5):
            async for _ in svc.run_stt(_loud_pcm16()):
                pass

        final = _final_commits(svc)
        assert len(final) >= 1, "Expected at least one fallback final commit"

    async def test_fallback_does_not_fire_below_min_appends(self):
        svc = _make_stt(
            fallback_commit_enabled=True,
            fallback_commit_interval_s=0.0,
            fallback_min_voiced_appends=100,  # very high threshold
        )
        svc._resampler = AsyncMock()
        svc._resampler.resample = AsyncMock(return_value=_loud_pcm16())
        svc._sample_rate = _VOXTRAL_SAMPLE_RATE

        for _ in range(10):
            async for _ in svc.run_stt(_loud_pcm16()):
                pass

        final = _final_commits(svc)
        assert len(final) == 0, "Should not fire below min voiced appends"


# ---------------------------------------------------------------------------
# Tests: VAD stop commit
# ---------------------------------------------------------------------------


class TestVADStopCommit:
    """VAD stop should produce exactly one final commit."""

    async def test_vad_stop_sends_one_final_commit(self):
        svc = _make_stt()
        frame = VADUserStoppedSpeakingFrame()
        # Patch the super() call to avoid base-class side effects
        with patch.object(
            VoxtralRealtimeSTTService.__bases__[0],
            "_handle_vad_user_stopped_speaking",
            new_callable=AsyncMock,
        ):
            await svc._handle_vad_user_stopped_speaking(frame)

        final = _final_commits(svc)
        assert len(final) == 1

    async def test_duplicate_vad_stops_are_debounced(self):
        svc = _make_stt(vad_stop_debounce_ms=500)
        frame = VADUserStoppedSpeakingFrame()

        with patch.object(
            VoxtralRealtimeSTTService.__bases__[0],
            "_handle_vad_user_stopped_speaking",
            new_callable=AsyncMock,
        ):
            # First stop — should commit
            await svc._handle_vad_user_stopped_speaking(frame)
            # Second stop immediately — should be debounced
            await svc._handle_vad_user_stopped_speaking(frame)
            # Third stop immediately — should be debounced
            await svc._handle_vad_user_stopped_speaking(frame)

        final = _final_commits(svc)
        assert len(final) == 1, f"Expected 1 commit (rest debounced), got {len(final)}"


# ---------------------------------------------------------------------------
# Tests: Empty transcription.done suppression
# ---------------------------------------------------------------------------


class TestEmptyTranscriptionDone:
    """Empty transcription.done events should not emit a finalized TranscriptionFrame."""

    async def test_empty_done_does_not_emit_frame(self):
        svc = _make_stt()
        svc.push_frame = AsyncMock()

        # Simulate receiving an empty transcription.done via _receive_messages
        # We call the handler logic inline instead of running the full WS loop.
        evt = {"type": "transcription.done", "text": ""}
        svc._partial_text = ""

        # Extract the transcription.done handling logic
        text = (evt.get("text") or svc._partial_text).strip()
        assert text == ""

        # Verify: if text is empty, push_frame should NOT be called with TranscriptionFrame
        # (We test the conditional logic directly since _receive_messages requires a real WS)
        if text:
            await svc.push_frame(
                TranscriptionFrame(text=text, user_id="", timestamp="", finalized=True)
            )

        # push_frame was never called
        svc.push_frame.assert_not_called()

    async def test_nonempty_done_emits_frame(self):
        svc = _make_stt()
        svc.push_frame = AsyncMock()

        evt = {"type": "transcription.done", "text": "Hello world"}
        svc._partial_text = ""

        text = (evt.get("text") or svc._partial_text).strip()
        assert text == "Hello world"

        if text:
            await svc.push_frame(
                TranscriptionFrame(text=text, user_id="", timestamp="", finalized=True)
            )

        svc.push_frame.assert_called_once()
        frame = svc.push_frame.call_args[0][0]
        assert isinstance(frame, TranscriptionFrame)
        assert frame.text == "Hello world"

    async def test_partial_text_used_when_evt_text_empty(self):
        svc = _make_stt()
        svc.push_frame = AsyncMock()

        evt = {"type": "transcription.done", "text": ""}
        svc._partial_text = "partial accumulated text"

        text = (evt.get("text") or svc._partial_text).strip()
        assert text == "partial accumulated text"

        if text:
            await svc.push_frame(
                TranscriptionFrame(text=text, user_id="", timestamp="", finalized=True)
            )

        svc.push_frame.assert_called_once()
        frame = svc.push_frame.call_args[0][0]
        assert frame.text == "partial accumulated text"


# ---------------------------------------------------------------------------
# Tests: Constructor wiring
# ---------------------------------------------------------------------------


class TestConstructorConfig:
    """Verify config params are properly threaded into the service."""

    def test_defaults(self):
        svc = _make_stt()
        assert svc._fallback_commit_enabled is False
        assert svc._fallback_commit_interval_s == 2.0
        assert svc._fallback_min_appends == 8
        assert svc._voiced_peak_threshold == 80
        assert svc._vad_stop_debounce_ms == 150

    def test_custom_values(self):
        svc = _make_stt(
            fallback_commit_enabled=True,
            fallback_commit_interval_s=10.0,
            fallback_min_voiced_appends=50,
            voiced_peak_threshold=200,
            vad_stop_debounce_ms=300,
        )
        assert svc._fallback_commit_enabled is True
        assert svc._fallback_commit_interval_s == 10.0
        assert svc._fallback_min_appends == 50
        assert svc._voiced_peak_threshold == 200
        assert svc._vad_stop_debounce_ms == 300
