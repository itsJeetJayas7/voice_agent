"""Chatterbox TTS server — OpenAI-compatible HTTP API.

Exposes POST /v1/audio/speech for text-to-speech synthesis using
Chatterbox-Turbo running on GPU.  Returns WAV audio at 24 kHz.
"""

from __future__ import annotations

import io
import struct
import time
import wave

import numpy as np
import perth
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field


# ── Monkey-patch watermarker (native lib unavailable) ───────
class _DummyWatermarker:
    def apply_watermark(self, audio, **kwargs):
        return audio

    def detect_watermark(self, audio, **kwargs):
        return False


perth.PerthImplicitWatermarker = _DummyWatermarker

from chatterbox.tts import ChatterboxTTS  # noqa: E402

# ── App ─────────────────────────────────────────────────────
app = FastAPI(title="Chatterbox TTS Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model: ChatterboxTTS | None = None


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=3000)
    voice: str = "default"
    model: str = "chatterbox-turbo"
    response_format: str = "wav"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8


@app.on_event("startup")
async def load_model():
    global model
    print("Loading Chatterbox-Turbo on CUDA...")
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device="cuda")
    print(f"Chatterbox loaded in {time.time() - t0:.1f}s  (SR={model.sr})")


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "chatterbox-turbo", "object": "model"}],
    }


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if model is None:
        return Response(content="Model not loaded", status_code=503)

    t0 = time.time()
    wav = model.generate(
        req.input,
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
        temperature=req.temperature,
    )
    gen_time = time.time() - t0

    # Convert tensor to PCM16 samples
    if wav.dim() > 1:
        wav = wav.squeeze(0)
    samples = wav.detach().cpu().numpy()
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    pcm_bytes = pcm.tobytes()

    duration = len(samples) / model.sr
    print(f"TTS: '{req.input[:60]}' → {duration:.1f}s audio in {gen_time:.1f}s")

    if req.response_format == "pcm":
        # Raw PCM16 — used by Pipecat's OpenAITTSService
        return Response(
            content=pcm_bytes,
            media_type="audio/pcm",
            headers={"X-Generation-Time": f"{gen_time:.2f}"},
        )

    # Default: WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(model.sr)
        wf.writeframes(pcm_bytes)
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"X-Generation-Time": f"{gen_time:.2f}"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
