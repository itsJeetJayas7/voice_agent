"""Unit tests for streaming behavior in the TTS HTTP server."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import httpx
import pytest
from fastapi.responses import StreamingResponse

from voice_agent import tts_server


class _DummyModel:
    sr = 24000


class _StubRequest:
    def __init__(self) -> None:
        self._disconnected = False

    async def is_disconnected(self) -> bool:
        return self._disconnected

    def disconnect(self) -> None:
        self._disconnected = True


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch: pytest.MonkeyPatch):
    """Avoid model startup and use a lightweight stub for tests."""
    original_model = tts_server.model
    original_refs = dict(tts_server.references)

    monkeypatch.setattr(tts_server, "model", _DummyModel())
    tts_server.references.clear()

    yield

    monkeypatch.setattr(tts_server, "model", original_model)
    tts_server.references.clear()
    tts_server.references.update(original_refs)


@pytest.mark.asyncio
async def test_pcm_request_returns_streaming_response():
    req = tts_server.SpeechRequest(input="hello world", response_format="pcm")
    resp = await tts_server.speech(req, _StubRequest())
    assert isinstance(resp, StreamingResponse)
    assert resp.media_type == "audio/pcm"


@pytest.mark.asyncio
async def test_first_stream_chunk_arrives_before_synthesis_finishes(
    monkeypatch: pytest.MonkeyPatch,
):
    finished = threading.Event()

    def fake_iter_pcm_stream(req, audio_prompt_path, cancel_event, stats):
        del req, audio_prompt_path, cancel_event, stats
        time.sleep(0.05)
        yield b"\x01\x02" * 128
        time.sleep(0.2)
        yield b"\x03\x04" * 128
        finished.set()

    monkeypatch.setattr(tts_server, "_iter_pcm_stream", fake_iter_pcm_stream)

    req = tts_server.SpeechRequest(input="hello world", response_format="pcm")
    stream = tts_server._stream_pcm_response(
        req,
        _StubRequest(),
        request_id="req-stream-1",
        audio_prompt_path=None,
    )

    first = await anext(stream)
    assert first
    assert not finished.is_set()

    remaining = [chunk async for chunk in stream]
    assert sum(len(chunk) for chunk in remaining) > 0
    assert finished.is_set()


@pytest.mark.asyncio
async def test_stream_cancel_propagates_to_worker(
    monkeypatch: pytest.MonkeyPatch,
):
    cancel_seen = threading.Event()

    def fake_iter_pcm_stream(req, audio_prompt_path, cancel_event, stats):
        del req, audio_prompt_path, stats
        while True:
            if cancel_event.is_set():
                cancel_seen.set()
                return
            time.sleep(0.01)
            yield b"\x00\x00" * 160

    monkeypatch.setattr(tts_server, "_iter_pcm_stream", fake_iter_pcm_stream)

    req = tts_server.SpeechRequest(input="cancel me", response_format="pcm")
    stream = tts_server._stream_pcm_response(
        req,
        _StubRequest(),
        request_id="req-stream-2",
        audio_prompt_path=None,
    )

    first = await anext(stream)
    assert first

    await stream.aclose()
    await asyncio.sleep(0.2)
    assert cancel_seen.is_set()


@pytest.mark.asyncio
async def test_streaming_emits_latency_metadata_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    def fake_iter_pcm_stream(req, audio_prompt_path, cancel_event, stats):
        del req, audio_prompt_path, cancel_event, stats
        time.sleep(0.02)
        yield b"\x11\x22" * 200
        time.sleep(0.02)
        yield b"\x33\x44" * 200

    monkeypatch.setattr(tts_server, "_iter_pcm_stream", fake_iter_pcm_stream)
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    req = tts_server.SpeechRequest(input="log me", response_format="pcm")
    stream = tts_server._stream_pcm_response(
        req,
        _StubRequest(),
        request_id="req-stream-3",
        audio_prompt_path=None,
    )
    chunks = [chunk async for chunk in stream]
    assert chunks

    assert any("tts_stream_first_chunk" in rec.getMessage() for rec in caplog.records)

    completion_logs = [
        rec.getMessage()
        for rec in caplog.records
        if "tts_stream_complete" in rec.getMessage()
    ]
    assert completion_logs

    completion = completion_logs[-1]
    assert "first_byte_latency_ms" in completion
    assert "total_latency_ms" in completion
    assert "audio_duration_s" in completion


@pytest.mark.asyncio
async def test_http_endpoint_streams_pcm_bytes(monkeypatch: pytest.MonkeyPatch):
    def fake_iter_pcm_stream(req, audio_prompt_path, cancel_event, stats):
        del req, audio_prompt_path, cancel_event, stats
        yield b"\xAA\xBB" * 32
        yield b"\xCC\xDD" * 32

    monkeypatch.setattr(tts_server, "_iter_pcm_stream", fake_iter_pcm_stream)

    transport = httpx.ASGITransport(app=tts_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"input": "hello", "response_format": "pcm"}
        async with client.stream("POST", "/v1/audio/speech", json=payload) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("audio/pcm")
            body = b"".join([chunk async for chunk in resp.aiter_bytes()])
            assert len(body) >= 128
