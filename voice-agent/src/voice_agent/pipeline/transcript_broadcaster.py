"""Transcript broadcaster — publishes transcripts via LiveKit data channel.

Captures TranscriptionFrame (user speech) and LLMTextFrame (agent response)
events and publishes them as JSON messages that the frontend can display.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_agent.logging import get_logger

if TYPE_CHECKING:
    from pipecat.transports.livekit.transport import LiveKitTransport

logger = get_logger("pipeline.transcript_broadcaster")


class TranscriptBroadcaster(FrameProcessor):
    """Intercepts transcript-related frames and publishes them via data channel."""

    def __init__(
        self,
        transport: LiveKitTransport,
        *,
        capture_user: bool = True,
        capture_agent: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._transport = transport
        self._capture_user = capture_user
        self._capture_agent = capture_agent
        self._agent_text_buffer = ""

    async def _publish(self, data: dict) -> None:
        """Publish a JSON message via the LiveKit data channel."""
        try:
            msg = json.dumps(data, ensure_ascii=False)
            await self._transport.send_message(msg)
        except Exception as e:
            logger.warning("Failed to publish transcript: %s", e)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # User speech — interim (partial)
        if self._capture_user and isinstance(frame, InterimTranscriptionFrame):
            await self._publish({
                "type": "transcript",
                "speaker": "user",
                "text": frame.text,
                "is_partial": True,
            })

        # User speech — final
        elif self._capture_user and isinstance(frame, TranscriptionFrame):
            await self._publish({
                "type": "transcript",
                "speaker": "user",
                "text": frame.text,
                "is_partial": False,
            })

        # Agent LLM response — streaming tokens
        elif self._capture_agent and isinstance(frame, LLMTextFrame):
            self._agent_text_buffer += frame.text

        # Agent LLM response — end of response
        elif self._capture_agent and isinstance(frame, LLMFullResponseEndFrame):
            if self._agent_text_buffer:
                await self._publish({
                    "type": "transcript",
                    "speaker": "agent",
                    "text": self._agent_text_buffer,
                    "is_partial": False,
                })
                self._agent_text_buffer = ""

        # Always pass frames downstream
        await self.push_frame(frame, direction)
