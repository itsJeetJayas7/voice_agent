"""Tests for the interruption controller."""

import asyncio

import pytest

from voice_agent.pipeline.interruption_controller import InterruptionController
from voice_agent.session import Session


@pytest.fixture
def session():
    return Session(room="test-room", identity="test-agent")


@pytest.fixture
def controller(session: Session):
    return InterruptionController(session)


class TestBargeIn:
    """Test barge-in interruption handling."""

    async def test_barge_in_when_speaking(
        self, controller: InterruptionController, session: Session
    ):
        """Barge-in should cancel turn when agent is speaking."""
        controller.agent_started_speaking()
        assert controller.is_agent_speaking
        assert controller.output_allowed

        await controller.handle_barge_in()

        assert not controller.is_agent_speaking
        assert not controller.output_allowed
        assert session.is_cancelled

    async def test_barge_in_when_not_speaking_is_noop(
        self, controller: InterruptionController, session: Session
    ):
        """Barge-in is a no-op if agent is not speaking."""
        await controller.handle_barge_in()
        assert not session.is_cancelled

    async def test_output_gate_closes_on_barge_in(
        self, controller: InterruptionController
    ):
        """Output gate must close immediately on barge-in."""
        controller.agent_started_speaking()
        assert controller.output_allowed

        await controller.handle_barge_in()
        assert not controller.output_allowed


class TestOutputControl:
    """Test output gate behaviour."""

    def test_output_allowed_by_default(self, controller: InterruptionController):
        assert controller.output_allowed

    def test_output_blocked_after_cancel(
        self, controller: InterruptionController, session: Session
    ):
        session.cancel_turn()
        assert not controller.output_allowed


class TestReset:
    """Test reset for new turn."""

    async def test_reset_restores_state(self, controller: InterruptionController):
        controller.agent_started_speaking()
        await controller.handle_barge_in()

        controller.reset_for_new_turn()
        assert not controller.is_agent_speaking
        assert controller.output_allowed


class TestTaskCancellation:
    """Test that barge-in cancels the right tasks."""

    async def test_tasks_cancelled_on_barge_in(
        self, controller: InterruptionController, session: Session
    ):
        """Named tasks should be cancelled on barge-in."""
        # Register mock tasks
        tts_task = asyncio.create_task(asyncio.sleep(100))
        llm_task = asyncio.create_task(asyncio.sleep(100))
        session.register_task("tts_task", tts_task)
        session.register_task("llm_task", llm_task)

        controller.agent_started_speaking()
        await controller.handle_barge_in()

        # Yield to event loop so tasks process their cancellation
        await asyncio.sleep(0)

        assert tts_task.cancelled()
        assert llm_task.cancelled()
