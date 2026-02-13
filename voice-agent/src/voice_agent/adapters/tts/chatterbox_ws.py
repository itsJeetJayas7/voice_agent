"""Chatterbox TTS adapter — WebSocket transport.

Streams text to the Chatterbox Turbo TTS service over a persistent
WebSocket and receives audio frames in real time.

NOTE: Request/response formats are marked with TODO(vendor-api)
where the actual Chatterbox WebSocket API schema must be mapped.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import websockets
import websockets.client

from voice_agent.adapters.tts.base import PCMFrame
from voice_agent.config import Settings
from voice_agent.logging import get_logger

logger = get_logger("tts.chatterbox_ws")


class ChatterboxWSAdapter:
    """WebSocket-based TTS adapter for Chatterbox Turbo.

    Implements the TTSAdapter protocol using a persistent WebSocket
    connection for lowest latency streaming synthesis.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ws_url = settings.chatterbox_ws_url
        self._timeout_s = settings.chatterbox_timeout_ms / 1000.0
        self._default_voice = settings.chatterbox_voice
        self._default_sr = settings.chatterbox_sample_rate
        self._ws: websockets.client.WebSocketClientProtocol | None = None  # type: ignore[name-defined]
        self._cancelled = False

    async def _ensure_connected(self) -> websockets.client.WebSocketClientProtocol:  # type: ignore[name-defined]
        if self._ws is None or self._ws.closed:
            # TODO(vendor-api): Add any required headers or auth tokens
            # for the Chatterbox WebSocket handshake.
            self._ws = await websockets.client.connect(  # type: ignore[attr-defined]
                self._ws_url,
                close_timeout=self._timeout_s,
                max_size=10 * 1024 * 1024,
            )
            self._cancelled = False
        return self._ws

    async def synthesize_stream(
        self,
        text_chunk: str,
        voice: str = "default",
        sample_rate: int = 24000,
    ) -> AsyncIterator[PCMFrame]:
        """Send text and yield PCM audio frames from the WebSocket."""
        if self._cancelled:
            return

        ws = await self._ensure_connected()
        voice = voice or self._default_voice
        sample_rate = sample_rate or self._default_sr

        # TODO(vendor-api): Map the actual Chatterbox WebSocket message format.
        # Adjust field names to match the real API.
        request_msg = json.dumps({
            "type": "synthesize",
            "text": text_chunk,
            "voice": voice,
            "sample_rate": sample_rate,
            "format": "pcm16",
        })

        try:
            await ws.send(request_msg)

            # Read audio frames until we get an end-of-synthesis signal
            while not self._cancelled:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=self._timeout_s)
                except asyncio.TimeoutError:
                    logger.warning("ChatterboxWS recv timeout")
                    break

                if isinstance(raw, bytes):
                    # Binary frame = audio data
                    if not raw:
                        break  # Empty frame signals end
                    yield PCMFrame(
                        audio=raw,
                        sample_rate=sample_rate,
                        channels=1,
                        samples_per_channel=len(raw) // 2,
                    )
                elif isinstance(raw, str):
                    # JSON control message
                    # TODO(vendor-api): Parse actual control messages.
                    # Expected: {"type": "end"} or {"type": "error", ...}
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") == "end":
                            break
                        elif msg.get("type") == "error":
                            logger.error(
                                "ChatterboxWS error: %s", msg.get("message", "")
                            )
                            break
                    except json.JSONDecodeError:
                        pass

        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed:
            if not self._cancelled:
                logger.warning("ChatterboxWS connection closed during synthesis")
        except Exception as exc:
            logger.error("ChatterboxWS synthesis error: %s", exc)

    async def cancel(self) -> None:
        self._cancelled = True
        if self._ws and not self._ws.closed:
            try:
                # TODO(vendor-api): Send cancel message if supported
                await self._ws.send(json.dumps({"type": "cancel"}))
            except Exception:
                pass

    async def close(self) -> None:
        self._cancelled = True
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("ChatterboxWS closed")
