"""Tests for the Chatterbox TTS adapters."""


import pytest

from voice_agent.adapters.tts.chatterbox_http import ChatterboxHTTPAdapter
from voice_agent.adapters.tts.chatterbox_ws import ChatterboxWSAdapter
from voice_agent.config import Settings


@pytest.fixture
def settings():
    return Settings(
        cerebras_api_key="test-key",
        chatterbox_protocol="http",
        chatterbox_host="localhost",
        chatterbox_port=8001,
        chatterbox_path="/v1/audio/speech",
        chatterbox_timeout_ms=5000,
        chatterbox_voice="default",
        chatterbox_sample_rate=24000,
        livekit_url="ws://localhost:7880",
        livekit_api_key="devkey",
        livekit_api_secret="devsecret",
    )


class TestChatterboxHTTP:
    """Tests for the HTTP TTS adapter."""

    @pytest.fixture
    def adapter(self, settings):
        return ChatterboxHTTPAdapter(settings)

    async def test_cancel_sets_flag(self, adapter: ChatterboxHTTPAdapter):
        await adapter.cancel()
        assert adapter._cancelled

    async def test_close_cleans_up(self, adapter: ChatterboxHTTPAdapter):
        await adapter.close()
        assert adapter._cancelled
        assert adapter._session is None

    async def test_synthesize_after_cancel_returns_empty(
        self, adapter: ChatterboxHTTPAdapter
    ):
        """Synthesis after cancel yields nothing."""
        await adapter.cancel()
        frames = []
        async for frame in adapter.synthesize_stream("Hello"):
            frames.append(frame)
        assert len(frames) == 0


class TestChatterboxWS:
    """Tests for the WebSocket TTS adapter."""

    @pytest.fixture
    def adapter(self, settings):
        settings.chatterbox_protocol = "ws"
        return ChatterboxWSAdapter(settings)

    async def test_cancel_sets_flag(self, adapter: ChatterboxWSAdapter):
        await adapter.cancel()
        assert adapter._cancelled

    async def test_close_without_connect(self, adapter: ChatterboxWSAdapter):
        await adapter.close()
        assert adapter._cancelled

    async def test_synthesize_after_cancel_returns_empty(
        self, adapter: ChatterboxWSAdapter
    ):
        """Synthesis after cancel yields nothing."""
        await adapter.cancel()
        frames = []
        async for frame in adapter.synthesize_stream("Hello"):
            frames.append(frame)
        assert len(frames) == 0
