"""Tests for the Voxtral STT adapters."""


import pytest

from voice_agent.adapters.stt.voxtral_http import VoxtralHTTPAdapter
from voice_agent.adapters.stt.voxtral_ws import VoxtralWSAdapter
from voice_agent.adapters.stt.base import STTEventType
from voice_agent.config import Settings


@pytest.fixture
def settings():
    return Settings(
        cerebras_api_key="test-key",
        voxtral_protocol="http",
        voxtral_host="localhost",
        voxtral_port=8000,
        voxtral_path="/v1/audio/transcriptions",
        voxtral_timeout_ms=5000,
        livekit_url="ws://localhost:7880",
        livekit_api_key="devkey",
        livekit_api_secret="devsecret",
    )


class TestVoxtralHTTP:
    """Tests for the HTTP STT adapter."""

    @pytest.fixture
    def adapter(self, settings):
        return VoxtralHTTPAdapter(settings)

    async def test_cancel_clears_buffer(self, adapter: VoxtralHTTPAdapter):
        """Cancel should clear the audio buffer."""
        await adapter.send_audio(b"\x00" * 1000)
        assert len(adapter._audio_buffer) > 0

        await adapter.cancel()
        assert len(adapter._audio_buffer) == 0
        assert adapter._cancelled

    async def test_send_after_cancel_is_noop(self, adapter: VoxtralHTTPAdapter):
        """Sending audio after cancel does nothing."""
        await adapter.cancel()
        await adapter.send_audio(b"\x00" * 100)
        assert len(adapter._audio_buffer) == 0

    async def test_close_cleans_up(self, adapter: VoxtralHTTPAdapter):
        """Close should clean up resources."""
        await adapter.connect()
        await adapter.close()
        assert adapter._session is None
        assert adapter._cancelled

    async def test_end_utterance_without_connect_emits_error(
        self, adapter: VoxtralHTTPAdapter
    ):
        """End utterance without connection emits error event."""
        adapter._audio_buffer.extend(b"\x00" * 100)
        await adapter.end_utterance()

        event = adapter._event_queue.get_nowait()
        assert event.type == STTEventType.ERROR

    async def test_buffer_accumulates(self, adapter: VoxtralHTTPAdapter):
        """Audio buffer should accumulate sent chunks."""
        await adapter.send_audio(b"\x00" * 100)
        await adapter.send_audio(b"\xff" * 100)
        assert len(adapter._audio_buffer) == 200


class TestVoxtralWS:
    """Tests for the WebSocket STT adapter."""

    @pytest.fixture
    def adapter(self, settings):
        settings.voxtral_protocol = "ws"
        return VoxtralWSAdapter(settings)

    async def test_cancel_sets_flag(self, adapter: VoxtralWSAdapter):
        """Cancel sets the cancelled flag."""
        await adapter.cancel()
        assert adapter._cancelled

    async def test_close_without_connect(self, adapter: VoxtralWSAdapter):
        """Close without prior connect should not raise."""
        await adapter.close()
        assert adapter._cancelled

    async def test_send_without_connect_is_noop(self, adapter: VoxtralWSAdapter):
        """Sending audio without connection does nothing."""
        await adapter.send_audio(b"\x00" * 100)
        # Should not raise
