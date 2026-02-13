"""Voxtral STT adapter — HTTP transport.

Sends audio chunks to the Voxtral STT service via HTTP POST and
receives transcript responses.

NOTE: Request/response formats are marked with TODO(vendor-api)
where the actual Voxtral API schema must be mapped.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import aiohttp

from voice_agent.adapters.stt.base import STTEvent, STTEventType
from voice_agent.config import Settings
from voice_agent.logging import get_logger

logger = get_logger("stt.voxtral_http")


class VoxtralHTTPAdapter:
    """HTTP-based STT adapter for Voxtral-Mini-4B-Realtime.

    Implements the STTAdapter protocol using HTTP POST requests.
    Audio is accumulated and sent in chunks; partial results are
    not available in pure HTTP mode (only final transcripts).
    For real-time partials, prefer the WebSocket transport.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.voxtral_base_url
        self._path = settings.voxtral_path
        self._timeout_s = settings.voxtral_timeout_ms / 1000.0
        self._session: aiohttp.ClientSession | None = None
        self._event_queue: asyncio.Queue[STTEvent] = asyncio.Queue()
        self._audio_buffer = bytearray()
        self._cancelled = False
        self._sample_rate = 16000
        self._channels = 1

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout_s)
        )
        self._cancelled = False
        logger.info("VoxtralHTTP connected to %s", self._base_url)

    async def send_audio(
        self,
        pcm16_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        if self._cancelled:
            return
        self._sample_rate = sample_rate
        self._channels = channels
        self._audio_buffer.extend(pcm16_bytes)

    async def end_utterance(self) -> None:
        """Send accumulated audio to Voxtral and emit a final transcript event."""
        if self._cancelled or not self._audio_buffer:
            return

        if not self._session:
            await self._event_queue.put(
                STTEvent(type=STTEventType.ERROR, error="Not connected")
            )
            return

        try:
            audio_data = bytes(self._audio_buffer)
            self._audio_buffer.clear()

            # TODO(vendor-api): Map the actual Voxtral HTTP API request format.
            # The payload structure below is a reasonable guess based on
            # OpenAI-compatible audio transcription endpoints.
            # Adjust Content-Type, field names, and response parsing
            # to match the real Voxtral API.
            url = f"{self._base_url}{self._path}"

            form = aiohttp.FormData()
            form.add_field(
                "file",
                audio_data,
                filename="audio.pcm",
                content_type="audio/pcm",
            )
            form.add_field("model", "voxtral-mini-4b-realtime-2602")
            form.add_field("language", "en")
            form.add_field("response_format", "json")
            # TODO(vendor-api): Add sample_rate and encoding fields if required.

            async with self._session.post(url, data=form) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    await self._event_queue.put(
                        STTEvent(
                            type=STTEventType.ERROR,
                            error=f"HTTP {resp.status}: {error_text}",
                        )
                    )
                    return

                # TODO(vendor-api): Parse the actual Voxtral response JSON.
                # Expected shape (OpenAI-compatible):
                #   {"text": "...", "language": "en", "segments": [...]}
                data = await resp.json()
                text = data.get("text", "")
                language = data.get("language", "")

                if text:
                    await self._event_queue.put(
                        STTEvent(
                            type=STTEventType.FINAL,
                            text=text,
                            language=language,
                            is_final=True,
                            confidence=1.0,
                        )
                    )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("VoxtralHTTP error: %s", exc)
            await self._event_queue.put(
                STTEvent(type=STTEventType.ERROR, error=str(exc))
            )

    async def events(self) -> AsyncIterator[STTEvent]:
        """Yield STT events as they become available."""
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
        self._audio_buffer.clear()
        # Drain the event queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def close(self) -> None:
        self._cancelled = True
        self._audio_buffer.clear()
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("VoxtralHTTP closed")
