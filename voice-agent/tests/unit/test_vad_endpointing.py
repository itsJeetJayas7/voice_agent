"""Tests for VAD + endpointing processor."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from voice_agent.pipeline.vad_endpointing import (
    EndpointingConfig,
    EndpointingProcessor,
    SpeechState,
)


@pytest.fixture
def config():
    return EndpointingConfig(
        vad_confidence=0.5,
        vad_start_secs=0.1,
        vad_stop_secs=0.5,
        vad_min_volume=0.001,
        endpoint_min_silence_ms=100,
        endpoint_max_silence_ms=500,
        max_utterance_secs=5,
    )


@pytest.fixture
def processor(config):
    return EndpointingProcessor(
        config=config,
        on_speech_start=AsyncMock(),
        on_speech_end=AsyncMock(),
        on_max_utterance=AsyncMock(),
    )


class TestSpeechStartDetection:
    """Test speech start state transitions."""

    async def test_initial_state_is_idle(self, processor: EndpointingProcessor):
        assert processor.state == SpeechState.IDLE

    async def test_speech_starts_tentatively(self, processor: EndpointingProcessor):
        await processor.process_vad_event(is_speech=True, confidence=0.8)
        assert processor.state == SpeechState.SPEECH_STARTED

    async def test_speech_confirms_after_start_secs(self, processor: EndpointingProcessor):
        await processor.process_vad_event(is_speech=True, confidence=0.8)
        assert processor.state == SpeechState.SPEECH_STARTED

        # Wait for start_secs to elapse
        await asyncio.sleep(0.15)  # > 0.1s
        await processor.process_vad_event(is_speech=True, confidence=0.8)
        assert processor.state == SpeechState.SPEECH_ACTIVE
        processor.on_speech_start.assert_awaited_once()

    async def test_low_confidence_ignored(self, processor: EndpointingProcessor):
        await processor.process_vad_event(is_speech=True, confidence=0.1)
        assert processor.state == SpeechState.IDLE


class TestSpeechEndDetection:
    """Test speech end / silence detection transitions."""

    async def _activate_speech(self, processor: EndpointingProcessor):
        """Helper to bring processor to SPEECH_ACTIVE state."""
        await processor.process_vad_event(is_speech=True, confidence=0.8)
        await asyncio.sleep(0.15)
        await processor.process_vad_event(is_speech=True, confidence=0.8)
        assert processor.state == SpeechState.SPEECH_ACTIVE

    async def test_silence_after_speech(self, processor: EndpointingProcessor):
        await self._activate_speech(processor)
        await processor.process_vad_event(is_speech=False)
        assert processor.state == SpeechState.SILENCE_AFTER_SPEECH

    async def test_speech_end_after_max_silence(self, processor: EndpointingProcessor):
        await self._activate_speech(processor)
        await processor.process_vad_event(is_speech=False)

        # Wait for max_silence to elapse
        await asyncio.sleep(0.55)  # > 500ms
        await processor.process_vad_event(is_speech=False)
        assert processor.state == SpeechState.IDLE
        processor.on_speech_end.assert_awaited_once()

    async def test_speech_resumes_cancels_endpoint(self, processor: EndpointingProcessor):
        await self._activate_speech(processor)
        await processor.process_vad_event(is_speech=False)
        assert processor.state == SpeechState.SILENCE_AFTER_SPEECH

        # Resume speech before silence threshold
        await processor.process_vad_event(is_speech=True, confidence=0.8)
        assert processor.state == SpeechState.SPEECH_ACTIVE
        processor.on_speech_end.assert_not_awaited()


class TestSTTFinalIntegration:
    """Test STT final signal integration with endpointing."""

    async def _get_to_silence(self, processor: EndpointingProcessor):
        await processor.process_vad_event(is_speech=True, confidence=0.8)
        await asyncio.sleep(0.15)
        await processor.process_vad_event(is_speech=True, confidence=0.8)
        await processor.process_vad_event(is_speech=False)

    async def test_stt_final_with_silence_endpoints_early(
        self, processor: EndpointingProcessor
    ):
        await self._get_to_silence(processor)
        assert processor.state == SpeechState.SILENCE_AFTER_SPEECH

        # STT final + min_silence should endpoint immediately
        await asyncio.sleep(0.12)  # > 100ms min_silence
        await processor.notify_stt_final()
        # STT final during silence triggers immediate endpoint
        assert processor.state == SpeechState.IDLE
        processor.on_speech_end.assert_awaited_once()


class TestMaxUtterance:
    """Test max utterance duration cutoff."""

    async def test_max_utterance_cutoff(self):
        config = EndpointingConfig(
            vad_confidence=0.5,
            vad_start_secs=0.05,
            max_utterance_secs=1,
        )
        processor = EndpointingProcessor(
            config=config,
            on_speech_start=AsyncMock(),
            on_speech_end=AsyncMock(),
            on_max_utterance=AsyncMock(),
        )

        await processor.process_vad_event(is_speech=True, confidence=0.8)
        await asyncio.sleep(0.06)
        await processor.process_vad_event(is_speech=True, confidence=0.8)

        # Simulate continuous speech for > max_utterance_secs
        await asyncio.sleep(1.1)
        await processor.process_vad_event(is_speech=True, confidence=0.8)

        processor.on_max_utterance.assert_awaited_once()


class TestReset:
    """Test reset functionality."""

    async def test_reset_returns_to_idle(self, processor: EndpointingProcessor):
        await processor.process_vad_event(is_speech=True, confidence=0.8)
        processor.reset()
        assert processor.state == SpeechState.IDLE
        assert processor._speech_start_time == 0.0
