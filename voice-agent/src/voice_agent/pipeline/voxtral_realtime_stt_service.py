"""Voxtral realtime STT service for vLLM's /v1/realtime endpoint.

This service implements the websocket protocol that is currently exposed by
local vLLM Voxtral realtime servers (session.update + input_audio_buffer.* +
transcription.delta/done events).
"""

from __future__ import annotations

import base64
import json
from typing import AsyncGenerator
from urllib.parse import quote_plus

import websockets
from websockets.protocol import State

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601

from voice_agent.logging import get_logger

logger = get_logger("pipeline.voxtral_realtime_stt")

_VOXTRAL_SAMPLE_RATE = 16000


class VoxtralRealtimeSTTService(WebsocketSTTService):
    """Pipecat STT service for Voxtral realtime websocket protocol."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        model: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_model_name(model)
        self._ws_url = (
            f"ws://{host}:{port}/v1/realtime?model={quote_plus(model)}"
        )
        self._receive_task = None
        self._session_ready = False
        self._partial_text = ""
        self._resampler = create_stream_resampler()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await self._send_json({"type": "input_audio_buffer.commit", "final": True})
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._session_ready:
            yield None
            return

        pcm16 = await self._resampler.resample(
            audio, self.sample_rate, _VOXTRAL_SAMPLE_RATE
        )
        if not pcm16:
            yield None
            return

        payload = base64.b64encode(pcm16).decode("utf-8")
        await self._send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": payload,
            }
        )
        yield None

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        await super()._handle_vad_user_stopped_speaking(frame)
        # Final commit closes the current transcription turn and triggers
        # transcription.done from vLLM.
        await self._send_json({"type": "input_audio_buffer.commit", "final": True})

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error),
                name="voxtral_realtime_receive",
            )

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task, timeout=1.0)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        if self._websocket and self._websocket.state is State.OPEN:
            return

        self._session_ready = False
        self._partial_text = ""
        self._websocket = await websockets.connect(self._ws_url)
        await self._call_event_handler("on_connected")
        logger.info("Connected Voxtral realtime STT at %s", self._ws_url)

    async def _disconnect_websocket(self):
        try:
            self._session_ready = False
            if self._websocket:
                await self._websocket.close()
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _send_json(self, payload: dict) -> None:
        try:
            if not self._disconnecting and self._websocket:
                await self._websocket.send(json.dumps(payload))
        except Exception as exc:
            if self._disconnecting or not self._websocket:
                return
            await self.push_error(error_msg=f"Voxtral STT send failed: {exc}", exception=exc)

    async def _receive_messages(self):
        async for raw in self._websocket:
            try:
                evt = json.loads(raw)
            except Exception:
                logger.debug("Skipping non-JSON STT WS message: %s", raw)
                continue

            if not isinstance(evt, dict):
                logger.debug("Skipping non-object STT WS event: %r", evt)
                continue

            evt_type = evt.get("type", "")

            if evt_type == "session.created":
                self._session_ready = True
                # vLLM's realtime endpoint expects this minimal update shape.
                await self._send_json({"type": "session.update", "model": self.model_name})
                # Prime the session; this matches the known-good client flow.
                await self._send_json({"type": "input_audio_buffer.commit"})

            elif evt_type == "session.updated":
                logger.debug("STT session updated")

            elif evt_type == "transcription.delta":
                delta = evt.get("delta", "")
                if delta:
                    self._partial_text += delta
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            text=self._partial_text,
                            user_id=self._user_id,
                            timestamp=time_now_iso8601(),
                        )
                    )

            elif evt_type == "transcription.done":
                text = (evt.get("text") or self._partial_text).strip()
                if text:
                    await self.push_frame(
                        TranscriptionFrame(
                            text=text,
                            user_id=self._user_id,
                            timestamp=time_now_iso8601(),
                            finalized=True,
                        )
                    )
                self._partial_text = ""
                # Rearm next turn after a finalized commit.
                await self._send_json({"type": "input_audio_buffer.commit"})

            elif evt_type == "error":
                message = evt.get("message") or evt.get("error") or str(evt)
                await self.push_error(error_msg=f"Voxtral realtime error: {message}")

            else:
                logger.debug("Unhandled STT WS event: %s", evt_type)
