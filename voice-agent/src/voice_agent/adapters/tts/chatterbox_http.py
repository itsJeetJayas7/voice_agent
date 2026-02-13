"""Chatterbox TTS adapter — HTTP transport.

Sends text chunks to the Chatterbox Turbo TTS service via HTTP POST
and receives synthesized audio.  Supports both streaming and
non-streaming response modes.

NOTE: Request/response formats are marked with TODO(vendor-api)
where the actual Chatterbox API schema must be mapped.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import aiohttp

from voice_agent.adapters.tts.base import PCMFrame
from voice_agent.config import Settings
from voice_agent.logging import get_logger

logger = get_logger("tts.chatterbox_http")


class ChatterboxHTTPAdapter:
    """HTTP-based TTS adapter for Chatterbox Turbo.

    Implements the TTSAdapter protocol.  If the server returns audio
    in one shot, splits it into 20ms frames for responsive playback.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.chatterbox_base_url
        self._path = settings.chatterbox_path
        self._timeout_s = settings.chatterbox_timeout_ms / 1000.0
        self._default_voice = settings.chatterbox_voice
        self._default_sr = settings.chatterbox_sample_rate
        self._session: aiohttp.ClientSession | None = None
        self._cancelled = False

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout_s)
            )
        return self._session

    async def synthesize_stream(
        self,
        text_chunk: str,
        voice: str = "default",
        sample_rate: int = 24000,
    ) -> AsyncIterator[PCMFrame]:
        """Synthesize text and yield PCM16 audio frames."""
        if self._cancelled:
            return

        session = await self._ensure_session()
        voice = voice or self._default_voice
        sample_rate = sample_rate or self._default_sr

        url = f"{self._base_url}{self._path}"

        # TODO(vendor-api): Map the actual Chatterbox HTTP API request format.
        # The payload structure below is a reasonable guess.
        # Adjust field names, content type, and response handling to match
        # the real Chatterbox API.
        payload = {
            "text": text_chunk,
            "voice": voice,
            "sample_rate": sample_rate,
            "format": "pcm16",
            # TODO(vendor-api): Add any additional params (speed, pitch, etc.)
        }

        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("ChatterboxHTTP error %d: %s", resp.status, error_text)
                    return

                # TODO(vendor-api): Determine if Chatterbox returns:
                # a) Streaming chunked audio (Transfer-Encoding: chunked)
                # b) Complete audio in one response
                # Handle both cases:

                content_type = resp.headers.get("Content-Type", "")
                if "audio" in content_type or resp.content_length:
                    # Non-streaming: read full audio and split into frames
                    full_audio = await resp.read()
                    if self._cancelled:
                        return

                    # Split into 20ms frames for responsive playback
                    frame_samples = (sample_rate * 20) // 1000  # 20ms
                    frame_bytes = frame_samples * 2  # 16-bit = 2 bytes per sample

                    for offset in range(0, len(full_audio), frame_bytes):
                        if self._cancelled:
                            return
                        chunk = full_audio[offset : offset + frame_bytes]
                        yield PCMFrame(
                            audio=chunk,
                            sample_rate=sample_rate,
                            channels=1,
                            samples_per_channel=len(chunk) // 2,
                        )
                else:
                    # Streaming: read chunks as they arrive
                    async for chunk in resp.content.iter_chunked(
                        (sample_rate * 20 * 2) // 1000  # 20ms of PCM16
                    ):
                        if self._cancelled:
                            return
                        yield PCMFrame(
                            audio=chunk,
                            sample_rate=sample_rate,
                            channels=1,
                            samples_per_channel=len(chunk) // 2,
                        )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("ChatterboxHTTP synthesis error: %s", exc)

    async def cancel(self) -> None:
        self._cancelled = True

    async def close(self) -> None:
        self._cancelled = True
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("ChatterboxHTTP closed")
