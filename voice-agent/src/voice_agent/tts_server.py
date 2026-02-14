"""Chatterbox TTS server — OpenAI-compatible HTTP API.

Exposes POST /v1/audio/speech for text-to-speech synthesis using
Chatterbox-Turbo running on GPU.  Returns WAV audio at 24 kHz.
"""

from __future__ import annotations

import io
import time
import uuid
import wave
from pathlib import Path as FSPath

import numpy as np
import perth
import librosa
from fastapi import FastAPI, HTTPException, UploadFile, File
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

from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402

# ── App ─────────────────────────────────────────────────────
app = FastAPI(title="Chatterbox TTS Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model: ChatterboxTurboTTS | None = None

REFERENCE_DIR = FSPath("/tmp/tts-references")
references: dict[str, dict] = {}  # ref_id -> {path, filename, duration}


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=3000)
    voice: str = "default"
    model: str = "chatterbox-turbo"
    response_format: str = "wav"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    reference_id: str | None = None


def _resolve_turbo_checkpoint() -> str:
    """Locate cached chatterbox-turbo weights, download if needed."""
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    repo = "ResembleAI/chatterbox-turbo"
    sentinel = "t3_turbo_v1.safetensors"

    # Fast path: already cached
    cached = try_to_load_from_cache(repo, sentinel)
    if cached and isinstance(cached, str):
        return str(FSPath(cached).parent)

    # Download (no token required for public repo)
    return snapshot_download(
        repo_id=repo,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
    )


@app.on_event("startup")
async def load_model():
    global model
    print("Loading Chatterbox-Turbo on CUDA...")
    t0 = time.time()
    ckpt_dir = _resolve_turbo_checkpoint()
    print(f"Using checkpoint: {ckpt_dir}")
    model = ChatterboxTurboTTS.from_local(ckpt_dir, device="cuda")

    # Patch: norm_loudness returns float64 (via pyloudnorm scalar math),
    # but s3tokenizer expects float32.  Cast output back to float32.
    _orig_norm = model.norm_loudness

    def _norm_f32(wav, sr, target_lufs=-27):
        out = _orig_norm(wav, sr, target_lufs)
        return out.astype(np.float32) if hasattr(out, "astype") else out

    model.norm_loudness = _norm_f32

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
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


@app.post("/v1/voice/reference")
async def upload_reference(file: UploadFile = File(...)):
    """Upload a reference audio clip for voice cloning."""
    allowed = {
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3",
        "audio/mp4", "audio/x-m4a", "audio/m4a",
        "audio/flac", "audio/x-flac",
    }
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported type: {file.content_type}. Use WAV, MP3, M4A, or FLAC.")

    contents = await file.read()
    max_bytes = 10 * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(400, f"File too large ({len(contents) / 1024 / 1024:.1f}MB). Max 10MB.")

    ref_id = uuid.uuid4().hex[:12]
    suffix = FSPath(file.filename).suffix if file.filename else ".wav"
    ref_path = REFERENCE_DIR / f"{ref_id}{suffix}"
    ref_path.write_bytes(contents)

    try:
        duration = librosa.get_duration(path=str(ref_path))
    except Exception:
        ref_path.unlink(missing_ok=True)
        raise HTTPException(400, "Cannot read audio file. Ensure it is a valid audio format.")

    references[ref_id] = {
        "path": str(ref_path),
        "filename": file.filename or "reference",
        "duration": round(duration, 1),
    }
    print(f"Reference uploaded: {file.filename} ({duration:.1f}s) → {ref_id}")
    return {"id": ref_id, "filename": file.filename, "duration_seconds": round(duration, 1)}


@app.delete("/v1/voice/reference/{ref_id}")
async def delete_reference(ref_id: str):
    """Delete a previously uploaded reference audio."""
    ref = references.pop(ref_id, None)
    if ref:
        FSPath(ref["path"]).unlink(missing_ok=True)
        print(f"Reference deleted: {ref_id}")
    return {"ok": True}


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if model is None:
        return Response(content="Model not loaded", status_code=503)

    audio_prompt_path = None
    if req.reference_id:
        ref = references.get(req.reference_id)
        if not ref:
            raise HTTPException(404, "Reference audio not found. Please re-upload.")
        audio_prompt_path = ref["path"]

    t0 = time.time()
    wav = model.generate(
        req.input,
        audio_prompt_path=audio_prompt_path,
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
