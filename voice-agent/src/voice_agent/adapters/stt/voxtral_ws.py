"""Voxtral STT adapter — WebSocket transport.

Streams audio to the Voxtral STT service over a persistent WebSocket
and receives partial/final transcript events in real time.

NOTE: Request/response formats are marked with TODO(vendor-api)
where the actual Voxtral WebSocket API schema must be mapped.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import websockets
import websockets.client

from voice_agent.adapters.stt.base import STTEvent, STTEventType
from voice_agent.config import Settings
from voice_agent.logging import get_logger

logger = get_logger("stt.voxtral_ws")


class VoxtralWSAdapter:
    """WebSocket-based STT adapter for Voxtral-Mini-4B-Realtime.

    Implements the STTAdapter protocol using a persistent WebSocket
    connection.  Supports incremental audio feeding with partial and
    final transcript events emitted in real time.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ws_url = settings.voxtral_ws_url
        self._timeout_s = settings.voxtral_timeout_ms / 1000.0
        self._ws: websockets.client.WebSocketClientProtocol | None = None  # type: ignore[name-defined]
        self._event_queue: asyncio.Queue[STTEvent] = asyncio.Queue()
        self._cancelled = False
        self._recv_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        # TODO(vendor-api): Add any required headers, auth tokens,
        # or query params for the Voxtral WebSocket handshake.
        self._ws = await websockets.client.connect(  # type: ignore[attr-defined]
            self._ws_url,
            close_timeout=self._timeout_s,
            max_size=10 * 1024 * 1024,  # 10MB max message
        )
        self._cancelled = False

        # Start background receiver task
        self._recv_task = asyncio.create_task(self._receive_loop())
        logger.info("VoxtralWS connected to %s", self._ws_url)

    async def _receive_loop(self) -> None:
        """Background loop reading messages from the WebSocket."""
        assert self._ws is not None
        try:
            async for raw_message in self._ws:
                if self._cancelled:
                    break

                try:
                    # TODO(vendor-api): Parse the actual Voxtral WebSocket
                    # message format.  The structure below is a reasonable
                    # guess.  Adjust field names and event types to match
                    # the real Voxtral streaming API.
                    #
                    # Expected message shapes:
                    # {"type": "partial", "text": "...", "confidence": 0.9}
                    # {"type": "final",   "text": "...", "language": "en"}
                    # {"type": "speech_start"}
                    # {"type": "speech_end"}
                    # {"type": "error",   "message": "..."}

                    if isinstance(raw_message, bytes):
                        # Binary frame — skip (might be server audio echo)
                        continue

                    msg = json.loads(raw_message)
                    msg_type = msg.get("type", "")

                    if msg_type == "partial":
                        await self._event_queue.put(
                            STTEvent(
                                type=STTEventType.PARTIAL,
                                text=msg.get("text", ""),
                                confidence=msg.get("confidence", 0.0),
                            )
                        )
                    elif msg_type == "final":
                        await self._event_queue.put(
                            STTEvent(
                                type=STTEventType.FINAL,
                                text=msg.get("text", ""),
                                language=msg.get("language", ""),
                                is_final=True,
                                confidence=msg.get("confidence", 1.0),
                            )
                        )
                    elif msg_type == "speech_start":
                        await self._event_queue.put(
                            STTEvent(type=STTEventType.SPEECH_START)
                        )
                    elif msg_type == "speech_end":
                        await self._event_queue.put(
                            STTEvent(type=STTEventType.SPEECH_END)
                        )
                    elif msg_type == "error":
                        await self._event_queue.put(
                            STTEvent(
                                type=STTEventType.ERROR,
                                error=msg.get("message", "Unknown error"),
                            )
                        )
                    else:
                        logger.debug("Unknown WS message type: %s", msg_type)

                except json.JSONDecodeError as exc:
                    logger.warning("Invalid JSON from STT WS: %s", exc)

        except websockets.exceptions.ConnectionClosed:
            if not self._cancelled:
                logger.warning("VoxtralWS connection closed unexpectedly")
                await self._event_queue.put(
                    STTEvent(type=STTEventType.ERROR, error="WebSocket closed")
                )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("VoxtralWS receive error: %s", exc)
            await self._event_queue.put(
                STTEvent(type=STTEventType.ERROR, error=str(exc))
            )

    async def send_audio(
        self,
        pcm16_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        if self._cancelled or not self._ws:
            return

        try:
            # TODO(vendor-api): Voxtral may expect audio in a specific
            # container or with a header.  For now, send raw PCM16 bytes
            # as a binary WebSocket frame.
            await self._ws.send(pcm16_bytes)
        except Exception as exc:
            logger.error("VoxtralWS send error: %s", exc)

    async def end_utterance(self) -> None:
        """Signal end of utterance to the STT service."""
        if self._cancelled or not self._ws:
            return

        try:
            # TODO(vendor-api): Send the appropriate end-of-utterance
            # signal.  Adjust JSON key/value as needed.
            await self._ws.send(json.dumps({"type": "end_utterance"}))
        except Exception as exc:
            logger.error("VoxtralWS end_utterance error: %s", exc)

    async def events(self) -> AsyncIterator[STTEvent]:
        """Yield STT events from the WebSocket receive loop."""
        while True:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=0.1
                )
                yield event
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def cancel(self) -> None:
        self._cancelled = True
        # Drain event queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def close(self) -> None:
        self._cancelled = True
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("VoxtralWS closed")
