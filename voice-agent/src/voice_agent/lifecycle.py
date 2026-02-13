"""Room lifecycle callbacks for LiveKit transport events."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from voice_agent.logging import get_logger

if TYPE_CHECKING:
    from pipecat.pipeline.task import PipelineTask

logger = get_logger("lifecycle")


async def on_first_participant_joined(
    transport: object,
    participant_id: str,
    task: PipelineTask,
) -> None:
    """Called when the first user joins the room.

    Sends a greeting after a short delay so the audio output is ready.
    """
    from pipecat.frames.frames import TTSSpeakFrame

    logger.info(
        "First participant joined: %s",
        participant_id,
        extra={"participant": participant_id, "event": "first_participant_joined"},
    )
    await asyncio.sleep(0.8)
    await task.queue_frame(
        TTSSpeakFrame("Hello! I'm here and ready to help. What would you like to talk about?")
    )


async def on_participant_left(
    transport: object,
    participant_id: str,
) -> None:
    """Called when a participant disconnects."""
    logger.info(
        "Participant left: %s",
        participant_id,
        extra={"participant": participant_id, "event": "participant_left"},
    )


async def on_participant_connected(
    transport: object,
    participant_id: str,
) -> None:
    """Called when a participant connects (including reconnects)."""
    logger.info(
        "Participant connected: %s",
        participant_id,
        extra={"participant": participant_id, "event": "participant_connected"},
    )
