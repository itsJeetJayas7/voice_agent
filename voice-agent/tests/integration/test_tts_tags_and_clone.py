"""Integration tests for gesture tag handling and voice clone regression.

Requires the TTS server running on localhost:8001 with ChatterboxTurboTTS.
Run: pytest tests/integration/test_tts_tags_and_clone.py -v -s
"""

from __future__ import annotations

import io
import wave

import pytest
import aiohttp

BASE_URL = "http://localhost:8001"


def _wav_duration(data: bytes) -> float:
    """Parse WAV header to get duration in seconds."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


@pytest.fixture
def session():
    """Create a client session scoped to the test."""
    import asyncio
    loop = asyncio.new_event_loop()
    sess = aiohttp.ClientSession()
    yield loop, sess
    loop.run_until_complete(sess.close())
    loop.close()


class TestHealthAndModels:
    """Sanity checks that the server is running."""

    @pytest.mark.asyncio
    async def test_health(self):
        async with aiohttp.ClientSession() as sess:
            async with sess.get(f"{BASE_URL}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "healthy"
                assert data["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_models_list(self):
        async with aiohttp.ClientSession() as sess:
            async with sess.get(f"{BASE_URL}/v1/models") as resp:
                assert resp.status == 200
                data = await resp.json()
                ids = [m["id"] for m in data["data"]]
                assert "chatterbox-turbo" in ids


class TestBasicSpeech:
    """Normal text generation still works."""

    @pytest.mark.asyncio
    async def test_plain_text_returns_wav(self):
        async with aiohttp.ClientSession() as sess:
            payload = {
                "input": "Hello, this is a basic test.",
                "model": "chatterbox-turbo",
                "response_format": "wav",
                "temperature": 0.8,
            }
            async with sess.post(f"{BASE_URL}/v1/audio/speech", json=payload) as resp:
                assert resp.status == 200
                assert resp.content_type == "audio/wav"
                data = await resp.read()
                assert len(data) > 1000, "Audio data too small"
                dur = _wav_duration(data)
                assert dur > 0.3, f"Audio too short: {dur:.2f}s"


class TestGestureTags:
    """Gesture tags are passed through and processed by the model."""

    @pytest.mark.asyncio
    async def test_tag_text_accepted(self):
        """Server accepts text containing [laugh] and returns audio."""
        async with aiohttp.ClientSession() as sess:
            payload = {
                "input": "Hello [laugh] there, how are you?",
                "model": "chatterbox-turbo",
                "response_format": "wav",
                "temperature": 0.8,
            }
            async with sess.post(f"{BASE_URL}/v1/audio/speech", json=payload) as resp:
                assert resp.status == 200
                data = await resp.read()
                assert len(data) > 1000
                dur = _wav_duration(data)
                assert dur > 0.3

    @pytest.mark.asyncio
    async def test_multiple_tags_accepted(self):
        """Multiple gesture tags in one request are accepted."""
        async with aiohttp.ClientSession() as sess:
            payload = {
                "input": "Oh [gasp] really? [laugh] That's funny [sigh].",
                "model": "chatterbox-turbo",
                "response_format": "wav",
                "temperature": 0.8,
            }
            async with sess.post(f"{BASE_URL}/v1/audio/speech", json=payload) as resp:
                assert resp.status == 200
                data = await resp.read()
                assert len(data) > 1000

    @pytest.mark.asyncio
    async def test_tag_only_accepted(self):
        """A request with only a gesture tag is accepted."""
        async with aiohttp.ClientSession() as sess:
            payload = {
                "input": "[laugh]",
                "model": "chatterbox-turbo",
                "response_format": "wav",
                "temperature": 0.8,
            }
            async with sess.post(f"{BASE_URL}/v1/audio/speech", json=payload) as resp:
                assert resp.status == 200
                data = await resp.read()
                assert len(data) > 500

    @pytest.mark.asyncio
    async def test_tag_changes_output_shape_vs_plain_text(self):
        """Tagged text should produce noticeably different timing than plain text."""
        async with aiohttp.ClientSession() as sess:
            plain_payload = {
                "input": "That is very funny.",
                "model": "chatterbox-turbo",
                "response_format": "wav",
                "temperature": 0.8,
            }
            tagged_payload = {
                "input": "That is very funny. [laugh]",
                "model": "chatterbox-turbo",
                "response_format": "wav",
                "temperature": 0.8,
            }

            async with sess.post(f"{BASE_URL}/v1/audio/speech", json=plain_payload) as resp:
                assert resp.status == 200
                plain_wav = await resp.read()

            async with sess.post(f"{BASE_URL}/v1/audio/speech", json=tagged_payload) as resp:
                assert resp.status == 200
                tagged_wav = await resp.read()

            plain_dur = _wav_duration(plain_wav)
            tagged_dur = _wav_duration(tagged_wav)
            assert tagged_dur > plain_dur + 0.2, (
                f"Expected tagged audio to be noticeably longer; "
                f"plain={plain_dur:.2f}s tagged={tagged_dur:.2f}s"
            )


class TestVoiceCloneRegression:
    """Voice cloning flow must still work after the model switch."""

    @pytest.fixture
    async def reference_id(self):
        """Upload a reference audio and return its ID. Clean up after test."""
        # Generate a reference audio first (using the server itself)
        async with aiohttp.ClientSession() as sess:
            gen_payload = {
                "input": "This is a reference sample for voice cloning testing. It needs to be longer than five seconds to work properly with the turbo model.",
                "model": "chatterbox-turbo",
                "response_format": "wav",
                "temperature": 0.8,
            }
            async with sess.post(f"{BASE_URL}/v1/audio/speech", json=gen_payload) as resp:
                assert resp.status == 200
                ref_audio = await resp.read()
                dur = _wav_duration(ref_audio)
                assert dur > 5.0, f"Reference audio too short ({dur:.1f}s), need >5s for Turbo"

            # Upload as reference
            form = aiohttp.FormData()
            form.add_field("file", ref_audio, filename="ref.wav", content_type="audio/wav")
            async with sess.post(f"{BASE_URL}/v1/voice/reference", data=form) as resp:
                assert resp.status == 200
                data = await resp.json()
                ref_id = data["id"]
                assert data["duration_seconds"] > 5.0

            yield ref_id

            # Cleanup
            await sess.delete(f"{BASE_URL}/v1/voice/reference/{ref_id}")

    @pytest.mark.asyncio
    async def test_clone_speech_works(self, reference_id):
        """Speech generation with a cloned voice reference returns audio."""
        async with aiohttp.ClientSession() as sess:
            payload = {
                "input": "This should sound like the cloned voice.",
                "model": "chatterbox-turbo",
                "response_format": "wav",
                "temperature": 0.8,
                "reference_id": reference_id,
            }
            async with sess.post(f"{BASE_URL}/v1/audio/speech", json=payload) as resp:
                assert resp.status == 200
                data = await resp.read()
                assert len(data) > 1000
                dur = _wav_duration(data)
                assert dur > 0.3

    @pytest.mark.asyncio
    async def test_clone_with_tags_works(self, reference_id):
        """Cloned voice + gesture tags both work simultaneously."""
        async with aiohttp.ClientSession() as sess:
            payload = {
                "input": "Hello [laugh] that's funny.",
                "model": "chatterbox-turbo",
                "response_format": "wav",
                "temperature": 0.8,
                "reference_id": reference_id,
            }
            async with sess.post(f"{BASE_URL}/v1/audio/speech", json=payload) as resp:
                assert resp.status == 200
                data = await resp.read()
                assert len(data) > 1000

    @pytest.mark.asyncio
    async def test_invalid_reference_returns_404(self):
        """Using a non-existent reference_id returns a clear error."""
        async with aiohttp.ClientSession() as sess:
            payload = {
                "input": "Test.",
                "model": "chatterbox-turbo",
                "reference_id": "nonexistent_id",
            }
            async with sess.post(f"{BASE_URL}/v1/audio/speech", json=payload) as resp:
                assert resp.status == 404
                data = await resp.json()
                assert "not found" in data["detail"].lower()
