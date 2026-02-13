"""Interruption controller — barge-in handling for immediate output cancellation.

On user speech start while agent is speaking, execute in order:
1. Stop sending audio frames to LiveKit output
2. Cancel TTS task
3. Cancel LLM streaming task
4. Reset turn state

Uses per-turn cancellation tokens and generation IDs to prevent
stale output from being emitted.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from voice_agent.logging import get_logger

if TYPE_CHECKING:
    from voice_agent.session import Session

logger = get_logger("interruption")


class InterruptionController:
    """Handles barge-in interruption for immediate output halt.

    Enforces true task cancellation — does not "ignore old output later"
    but actually cancels in-flight tasks and flushes output.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._is_agent_speaking = False
        self._output_gate_open = True

    @property
    def is_agent_speaking(self) -> bool:
        return self._is_agent_speaking

    @property
    def output_allowed(self) -> bool:
        """Check if audio output is currently allowed."""
        return self._output_gate_open and not self._session.is_cancelled

    def agent_started_speaking(self) -> None:
        """Mark that the agent has started producing audio."""
        self._is_agent_speaking = True
        self._output_gate_open = True

    def agent_stopped_speaking(self) -> None:
        """Mark that the agent has finished producing audio."""
        self._is_agent_speaking = False

    async def handle_barge_in(self, turn_metrics: object | None = None) -> None:
        """Handle user speech start during agent speech.

        Executes the interruption sequence:
        1. Close output gate (stop sending audio frames)
        2. Cancel TTS task
        3. Cancel LLM task
        4. Reset turn state

        Args:
            turn_metrics: Optional TurnMetrics to record barge-in timing.
        """
        if not self._is_agent_speaking:
            logger.debug("Barge-in ignored: agent not speaking")
            return

        barge_in_start = time.monotonic()
        logger.info(
            "Barge-in detected — interrupting agent",
            extra={
                "session_id": self._session.session_id,
                "turn_id": self._session.turn_id,
                "event": "barge_in",
            },
        )

        # Step 1: Close the output gate immediately
        self._output_gate_open = False
        self._is_agent_speaking = False

        # Step 2: Signal turn cancellation
        self._session.cancel_turn()

        # Step 3: Cancel TTS task
        self._session.cancel_tasks("tts_task")

        # Step 4: Cancel LLM task
        self._session.cancel_tasks("llm_task")

        # Step 5: Cancel chunker task
        self._session.cancel_tasks("chunker_task")

        barge_in_end = time.monotonic()

        # Record metrics
        if turn_metrics and hasattr(turn_metrics, "barge_in_speech_start_at"):
            turn_metrics.barge_in_speech_start_at = barge_in_start
            turn_metrics.barge_in_playback_halted_at = barge_in_end  # type: ignore[attr-defined]

        logger.info(
            "Barge-in completed in %.1fms",
            (barge_in_end - barge_in_start) * 1000,
            extra={
                "session_id": self._session.session_id,
                "event": "barge_in_complete",
            },
        )

    def reset_for_new_turn(self) -> None:
        """Reset interruption state for a new conversation turn."""
        self._is_agent_speaking = False
        self._output_gate_open = True
        self._session._cancel_event.clear()
