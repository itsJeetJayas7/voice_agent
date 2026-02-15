"""Voxtral realtime STT service for vLLM's /v1/realtime endpoint.

This service implements the websocket protocol that is currently exposed by
local vLLM Voxtral realtime servers (session.update + input_audio_buffer.* +
transcription.delta/done events).
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import AsyncGenerator
from urllib.parse import quote_plus

import numpy as np
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
        fallback_commit_enabled: bool = False,
        fallback_commit_interval_s: float = 5.0,
        fallback_min_voiced_appends: int = 20,
        voiced_peak_threshold: int = 80,
        vad_stop_debounce_ms: int = 150,
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
        self._append_count = 0
        self._commit_count = 0
        self._voiced_append_since_commit = 0
        self._last_commit_ts = 0.0
        # Fallback commit config (disabled by default to avoid premature cuts)
        self._fallback_commit_enabled = fallback_commit_enabled
        self._fallback_commit_interval_s = fallback_commit_interval_s
        self._fallback_min_appends = fallback_min_voiced_appends
        self._voiced_peak_threshold = voiced_peak_threshold
        # VAD stop debounce to prevent duplicate final commits
        self._vad_stop_debounce_ms = vad_stop_debounce_ms
        self._last_vad_stop_commit_ts = 0.0
        self._connect_retry_interval_s = 2.0
        self._connect_startup_timeout_s = 180.0

    async def start(self, frame: StartFrame):
        await super().start(frame)
        deadline = time.monotonic() + self._connect_startup_timeout_s
        while True:
            try:
                await self._connect()
                return
            except Exception as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                sleep_s = min(self._connect_retry_interval_s, remaining)
                logger.warning(
                    "STT connect failed, retrying in %.1fs (remaining=%.1fs): %s",
                    sleep_s,
                    remaining,
                    exc,
                )
                await asyncio.sleep(sleep_s)

    async def stop(self, frame: EndFrame):
        await self._commit_audio(reason="shutdown")
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
        self._append_count += 1
        samples = np.frombuffer(pcm16, dtype=np.int16)
        peak = int(np.max(np.abs(samples))) if samples.size else 0
        if peak >= self._voiced_peak_threshold:
            self._voiced_append_since_commit += 1
        if self._append_count == 1 or self._append_count % 100 == 0:
            logger.info(
                "STT append sent (count=%d, in_bytes=%d, out_bytes=%d, in_rate=%d, peak=%d)",
                self._append_count,
                len(audio),
                len(pcm16),
                self.sample_rate,
                peak,
            )

        # Fallback: if VAD stop frames are missed, force periodic final commits
        # so transcription can complete and the pipeline can progress.
        # Disabled by default — only fires when explicitly enabled.
        if self._fallback_commit_enabled:
            now = time.monotonic()
            if (
                self._voiced_append_since_commit >= self._fallback_min_appends
                and now - self._last_commit_ts >= self._fallback_commit_interval_s
            ):
                logger.info(
                    "STT fallback commit triggered (voiced_append_since_commit=%d, elapsed=%.2fs)",
                    self._voiced_append_since_commit,
                    now - self._last_commit_ts,
                )
                await self._commit_audio(reason="fallback")
        yield None

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        await super()._handle_vad_user_stopped_speaking(frame)
        # Debounce rapid duplicate VAD stop events
        now = time.monotonic()
        elapsed_ms = (now - self._last_vad_stop_commit_ts) * 1000
        if elapsed_ms < self._vad_stop_debounce_ms:
            logger.info(
                "VAD stop debounced (elapsed_ms=%.1f, debounce_ms=%d)",
                elapsed_ms,
                self._vad_stop_debounce_ms,
            )
            return
        # Final commit closes the current transcription turn and triggers
        # transcription.done from vLLM.
        logger.info("VAD stop received by STT service; committing audio buffer (final=true)")
        await self._commit_audio(reason="vad_stop")

    async def _commit_audio(self, *, reason: str) -> None:
        """Send input_audio_buffer.commit with final=true and log the reason."""
        logger.info(
            "STT commit (reason=%s, voiced_since_last=%d)",
            reason,
            self._voiced_append_since_commit,
        )
        await self._send_json({"type": "input_audio_buffer.commit", "final": True})
        if reason == "vad_stop":
            self._last_vad_stop_commit_ts = time.monotonic()

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
        self._append_count = 0
        self._commit_count = 0
        self._voiced_append_since_commit = 0
        self._last_commit_ts = time.monotonic()
        self._last_vad_stop_commit_ts = 0.0
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
                msg_type = payload.get("type", "unknown")
                if msg_type == "input_audio_buffer.commit":
                    self._commit_count += 1
                    self._voiced_append_since_commit = 0
                    self._last_commit_ts = time.monotonic()
                    logger.info(
                        "STT commit sent (count=%d, final=%s)",
                        self._commit_count,
                        payload.get("final"),
                    )
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
                logger.info("STT websocket session.created received")
                self._session_ready = True
                # vLLM's realtime endpoint expects this minimal update shape.
                await self._send_json({"type": "session.update", "model": self.model_name})
                # Prime the session; this matches the known-good client flow.
                await self._send_json({"type": "input_audio_buffer.commit"})

            elif evt_type == "session.updated":
                logger.info("STT websocket session.updated received")

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
                logger.info(
                    "STT transcription.done received (text_chars=%d, partial_chars=%d)",
                    len(evt.get("text") or ""),
                    len(self._partial_text),
                )
                if text:
                    await self.push_frame(
                        TranscriptionFrame(
                            text=text,
                            user_id=self._user_id,
                            timestamp=time_now_iso8601(),
                            finalized=True,
                        )
                    )
                else:
                    logger.debug("Suppressing empty transcription.done (no text to emit)")
                self._partial_text = ""
                # Rearm next turn after a finalized commit.
                await self._send_json({"type": "input_audio_buffer.commit"})

            elif evt_type == "error":
                message = evt.get("message") or evt.get("error") or str(evt)
                await self.push_error(error_msg=f"Voxtral realtime error: {message}")

            else:
                logger.debug("Unhandled STT WS event: %s", evt_type)
