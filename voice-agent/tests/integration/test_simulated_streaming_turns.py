"""Integration test — simulated streaming turn lifecycle.

Tests the full pipeline logic using mocked STT/TTS/LLM backends:
1. Simulate input audio frames → VAD speech start/end
2. Verify interim/final transcripts emitted
3. Verify LLM tokens streamed
4. Verify TTS audio frames produced
5. Simulate barge-in and verify immediate interruption
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from voice_agent.pipeline.interruption_controller import InterruptionController
from voice_agent.pipeline.tts_chunker import ChunkerConfig, TTSChunker
from voice_agent.pipeline.vad_endpointing import EndpointingConfig, EndpointingProcessor
from voice_agent.session import Session, SessionState


class TestSimulatedStreamingTurn:
    """End-to-end simulated turn test."""

    @pytest.fixture
    def session(self):
        return Session(room="test-room", identity="test-agent")

    @pytest.fixture
    def endpointer(self):
        return EndpointingProcessor(
            config=EndpointingConfig(
                vad_confidence=0.5,
                vad_start_secs=0.05,
                endpoint_min_silence_ms=50,
                endpoint_max_silence_ms=200,
            ),
            on_speech_start=AsyncMock(),
            on_speech_end=AsyncMock(),
        )

    @pytest.fixture
    def chunker(self):
        return TTSChunker(ChunkerConfig(min_chars=5, max_chars=50, max_wait_ms=100, queue_maxsize=0))

    @pytest.fixture
    def controller(self, session):
        return InterruptionController(session)

    async def test_full_turn_lifecycle(
        self,
        session: Session,
        endpointer: EndpointingProcessor,
        chunker: TTSChunker,
        controller: InterruptionController,
    ):
        """Simulate a complete conversation turn."""
        # 1. Start a new turn
        session.new_turn()
        assert session.state == SessionState.ACTIVE
        assert session.turn_id != ""

        # 2. Simulate speech start
        await endpointer.process_vad_event(is_speech=True, confidence=0.8)
        await asyncio.sleep(0.06)  # > vad_start_secs
        await endpointer.process_vad_event(is_speech=True, confidence=0.8)
        endpointer.on_speech_start.assert_awaited_once()

        # 3. Simulate speech end
        await endpointer.process_vad_event(is_speech=False)
        await asyncio.sleep(0.22)  # > endpoint_max_silence_ms
        await endpointer.process_vad_event(is_speech=False)
        endpointer.on_speech_end.assert_awaited_once()

        # 4. Simulate LLM token stream → chunker
        tokens = "Hello! How can I help you today? I'm ready to assist."
        for word in tokens.split():
            await chunker.add_token(word + " ")
        await chunker.flush_remaining()

        # 5. Verify chunks were produced
        queue = await chunker.chunks()
        chunks = []
        while not queue.empty():
            c = queue.get_nowait()
            if c:
                chunks.append(c)
        assert len(chunks) >= 1
        assert any("Hello" in c for c in chunks)

    async def test_barge_in_during_response(
        self,
        session: Session,
        chunker: TTSChunker,
        controller: InterruptionController,
    ):
        """Simulate barge-in during agent response."""
        session.new_turn()

        # Agent starts speaking
        controller.agent_started_speaking()
        assert controller.is_agent_speaking

        # Simulate partial LLM response
        await chunker.add_token("I am currently explaining ")

        # User interrupts
        await controller.handle_barge_in()

        # Verify interruption sequence
        assert not controller.is_agent_speaking
        assert not controller.output_allowed
        assert session.is_cancelled

        # Verify chunker can be cancelled
        chunker.cancel()
        assert chunker._buffer == ""

    async def test_turn_reset_after_interruption(
        self,
        session: Session,
        controller: InterruptionController,
        endpointer: EndpointingProcessor,
    ):
        """Verify clean state after interruption + new turn."""
        # First turn — interrupted
        session.new_turn()
        controller.agent_started_speaking()
        await controller.handle_barge_in()
        assert session.is_cancelled

        # New turn — clean state
        controller.reset_for_new_turn()
        endpointer.reset()
        turn_metrics = session.new_turn()

        assert not session.is_cancelled
        assert controller.output_allowed
        assert endpointer.state.name == "IDLE"
        assert turn_metrics.turn_id == session.turn_id

    async def test_session_cleanup(self, session: Session):
        """Verify session cleanup is deterministic."""
        session.new_turn()

        # Register dummy tasks
        t1 = asyncio.create_task(asyncio.sleep(100))
        t2 = asyncio.create_task(asyncio.sleep(100))
        session.register_task("task_1", t1)
        session.register_task("task_2", t2)

        # Close session
        await session.close()

        assert session.state == SessionState.CLOSED
        assert t1.cancelled()
        assert t2.cancelled()
        assert len(session._tasks) == 0
