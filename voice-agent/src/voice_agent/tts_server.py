"""Chatterbox TTS server — OpenAI-compatible HTTP API.

Exposes POST /v1/audio/speech for text-to-speech synthesis using
Chatterbox-Turbo running on GPU.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
import uuid
import wave
from collections.abc import Iterator
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path as FSPath

import librosa
import numpy as np
import perth
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


# ── Monkey-patch watermarker (native lib unavailable) ───────
class _DummyWatermarker:
    def apply_watermark(self, audio, **kwargs):
        return audio

    def detect_watermark(self, audio, **kwargs):
        return False


perth.PerthImplicitWatermarker = _DummyWatermarker

from chatterbox.models.s3gen.const import S3GEN_SIL  # noqa: E402
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm  # noqa: E402

# ── App ─────────────────────────────────────────────────────
app = FastAPI(title="Chatterbox TTS Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

model: ChatterboxTurboTTS | None = None
MODEL_LOCK = threading.Lock()

REFERENCE_DIR = FSPath("/tmp/tts-references")
references: dict[str, dict[str, object]] = {}

PCM_SAMPLE_WIDTH_BYTES = 2
PCM_STREAM_QUEUE_MAXSIZE = 8
PCM_STREAM_QUEUE_POLL_TIMEOUT_S = 0.05
PCM_STREAM_FRAME_MS = 20
STREAM_FIRST_DECODE_TOKENS = 4
STREAM_DECODE_STEP_TOKENS = 8
STREAM_MAX_GEN_TOKENS = 1000


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=3000)
    voice: str = "default"
    model: str = "chatterbox-turbo"
    response_format: str = "wav"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    reference_id: str | None = None


@dataclass
class _SynthesisStats:
    request_id: str
    request_started_at: float
    chars: int
    emitted_chunks: int = 0
    emitted_bytes: int = 0
    emitted_samples: int = 0
    generated_tokens: int = 0
    first_chunk_generated_at: float | None = None
    first_chunk_emitted_at: float | None = None
    synthesis_finished_at: float | None = None
    cancelled: bool = False
    backend: str = ""
    error: str | None = None


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


def _now() -> float:
    return time.perf_counter()


def _float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)


def _pcm_bytes_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _supports_incremental_streaming(tts_model: ChatterboxTurboTTS) -> bool:
    return (
        hasattr(tts_model, "t3")
        and hasattr(tts_model, "s3gen")
        and hasattr(tts_model, "tokenizer")
        and hasattr(tts_model.s3gen, "flow_inference")
        and hasattr(tts_model.s3gen, "hift_inference")
    )


def _iter_turbo_tokens(
    tts_model: ChatterboxTurboTTS,
    text: str,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    cancel_event: threading.Event,
) -> Iterator[int]:
    """Yield Turbo speech tokens incrementally.

    Mirrors chatterbox's inference_turbo logic but yields token ids as soon as
    they are sampled, enabling downstream incremental vocoding.
    """
    t3 = tts_model.t3

    text_tokens = tts_model.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).input_ids.to(tts_model.device)

    logits_processors = LogitsProcessorList()
    if temperature > 0 and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        logits_processors.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p))
    if repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

    embeds, _ = t3.prepare_input_embeds(
        t3_cond=tts_model.conds.t3,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )

    generated_speech_tokens: list[torch.Tensor] = []

    llm_outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    hidden_states = llm_outputs[0]
    past_key_values = llm_outputs.past_key_values

    speech_logits = t3.speech_head(hidden_states)
    processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
    if torch.all(processed_logits == -float("inf")):
        return

    probs = torch.softmax(processed_logits, dim=-1)
    next_speech_token = torch.multinomial(probs, num_samples=1)

    generated_speech_tokens.append(next_speech_token)
    current_speech_token = next_speech_token
    first_token_id = int(next_speech_token.item())
    if first_token_id != t3.hp.stop_speech_token:
        yield first_token_id

    max_gen_len = min(STREAM_MAX_GEN_TOKENS, int(getattr(t3.hp, "max_speech_tokens", STREAM_MAX_GEN_TOKENS)))
    for _ in range(max_gen_len):
        if cancel_event.is_set():
            return

        current_speech_embed = t3.speech_emb(current_speech_token)

        llm_outputs = t3.tfmr(
            inputs_embeds=current_speech_embed,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values

        speech_logits = t3.speech_head(hidden_states)
        input_ids = torch.cat(generated_speech_tokens, dim=1)
        processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])
        if torch.all(processed_logits == -float("inf")):
            return

        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)

        generated_speech_tokens.append(next_speech_token)
        current_speech_token = next_speech_token
        token_id = int(next_speech_token.item())
        if token_id == t3.hp.stop_speech_token:
            return

        yield token_id


def _decode_incremental_audio_delta(
    tts_model: ChatterboxTurboTTS,
    *,
    token_ids: list[int],
    emitted_samples: int,
    finalize: bool,
) -> bytes:
    if not token_ids:
        return b""

    speech_tokens = torch.tensor(token_ids, dtype=torch.long, device=tts_model.device)

    output_mels = tts_model.s3gen.flow_inference(
        speech_tokens=speech_tokens,
        ref_dict=tts_model.conds.gen,
        n_cfm_timesteps=2,
        finalize=finalize,
    )
    output_mels = output_mels.to(dtype=tts_model.s3gen.dtype)

    output_wavs, _ = tts_model.s3gen.hift_inference(output_mels, None)

    trim_fade = getattr(tts_model.s3gen, "trim_fade", None)
    if trim_fade is not None and output_wavs.size(1) > 0:
        fade_len = min(int(trim_fade.shape[0]), int(output_wavs.size(1)))
        output_wavs = output_wavs.clone()
        output_wavs[:, :fade_len] = output_wavs[:, :fade_len] * trim_fade[:fade_len]

    samples = output_wavs.squeeze(0).detach().cpu().numpy()
    pcm = _float_to_pcm16(samples)

    if emitted_samples >= pcm.shape[0]:
        return b""

    return pcm[emitted_samples:].tobytes()


def _iter_incremental_pcm_stream(
    req: SpeechRequest,
    audio_prompt_path: str | None,
    cancel_event: threading.Event,
    stats: _SynthesisStats,
) -> Iterator[bytes]:
    """Yield PCM chunks incrementally from Chatterbox Turbo internals."""
    assert model is not None

    if audio_prompt_path:
        model.prepare_conditionals(
            audio_prompt_path,
            exaggeration=req.exaggeration,
            norm_loudness=True,
        )
    elif model.conds is None:
        raise RuntimeError("No default conditionals loaded for Chatterbox Turbo")

    text = punc_norm(req.input)

    valid_tokens: list[int] = []
    emitted_samples = 0
    next_decode_threshold = STREAM_FIRST_DECODE_TOKENS

    for token_id in _iter_turbo_tokens(
        model,
        text,
        temperature=req.temperature,
        top_k=1000,
        top_p=0.95,
        repetition_penalty=1.2,
        cancel_event=cancel_event,
    ):
        if cancel_event.is_set():
            stats.cancelled = True
            return

        stats.generated_tokens += 1

        # Drop OOV/control tokens, consistent with chatterbox generate().
        if token_id >= 6561:
            continue

        valid_tokens.append(token_id)

        if len(valid_tokens) < next_decode_threshold:
            continue

        pcm_delta = _decode_incremental_audio_delta(
            model,
            token_ids=valid_tokens,
            emitted_samples=emitted_samples,
            # finalize=False currently throws a shape mismatch in the shipped
            # chatterbox package for short/partial token prefixes. Using
            # finalize=True keeps incremental decoding stable.
            finalize=True,
        )
        if pcm_delta:
            emitted_samples += len(pcm_delta) // PCM_SAMPLE_WIDTH_BYTES
            yield pcm_delta

        next_decode_threshold += STREAM_DECODE_STEP_TOKENS

    if cancel_event.is_set():
        stats.cancelled = True
        return

    if not valid_tokens:
        return

    # Final flush with explicit silence suffix, consistent with turbo generate().
    final_tokens = valid_tokens + [S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]
    pcm_delta = _decode_incremental_audio_delta(
        model,
        token_ids=final_tokens,
        emitted_samples=emitted_samples,
        finalize=True,
    )
    if pcm_delta:
        yield pcm_delta


def _iter_blocking_pcm_stream(
    req: SpeechRequest,
    audio_prompt_path: str | None,
    cancel_event: threading.Event,
) -> Iterator[bytes]:
    """Fallback stream: generate full waveform, then yield frame-sized PCM chunks."""
    assert model is not None

    wav = model.generate(
        req.input,
        audio_prompt_path=audio_prompt_path,
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
        temperature=req.temperature,
    )

    if cancel_event.is_set():
        return

    if wav.dim() > 1:
        wav = wav.squeeze(0)
    samples = wav.detach().cpu().numpy()
    pcm = _float_to_pcm16(samples)
    pcm_bytes = pcm.tobytes()

    frame_samples = (model.sr * PCM_STREAM_FRAME_MS) // 1000
    frame_bytes = frame_samples * PCM_SAMPLE_WIDTH_BYTES
    for offset in range(0, len(pcm_bytes), frame_bytes):
        if cancel_event.is_set():
            return
        yield pcm_bytes[offset : offset + frame_bytes]


def _iter_pcm_stream(
    req: SpeechRequest,
    audio_prompt_path: str | None,
    cancel_event: threading.Event,
    stats: _SynthesisStats,
) -> Iterator[bytes]:
    assert model is not None

    if _supports_incremental_streaming(model):
        stats.backend = "incremental_turbo"
        yield from _iter_incremental_pcm_stream(req, audio_prompt_path, cancel_event, stats)
        return

    stats.backend = "blocking_fallback"
    logger.warning("Incremental TTS backend unavailable; using blocking fallback stream")
    yield from _iter_blocking_pcm_stream(req, audio_prompt_path, cancel_event)


def _put_chunk_with_backpressure(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[bytes | None],
    chunk: bytes | None,
    cancel_event: threading.Event,
) -> bool:
    while not cancel_event.is_set():
        future = asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
        try:
            future.result(timeout=0.2)
            return True
        except FuturesTimeoutError:
            continue
        except Exception:
            return False
    return False


def _run_synthesis_worker(
    *,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[bytes | None],
    req: SpeechRequest,
    audio_prompt_path: str | None,
    cancel_event: threading.Event,
    stats: _SynthesisStats,
) -> None:
    try:
        with MODEL_LOCK:
            for pcm_chunk in _iter_pcm_stream(req, audio_prompt_path, cancel_event, stats):
                if cancel_event.is_set():
                    stats.cancelled = True
                    break

                if not pcm_chunk:
                    continue

                stats.emitted_chunks += 1
                stats.emitted_bytes += len(pcm_chunk)
                stats.emitted_samples += len(pcm_chunk) // PCM_SAMPLE_WIDTH_BYTES
                if stats.first_chunk_generated_at is None:
                    stats.first_chunk_generated_at = _now()

                ok = _put_chunk_with_backpressure(loop, queue, pcm_chunk, cancel_event)
                if not ok:
                    stats.cancelled = True
                    break
    except Exception as exc:
        stats.error = repr(exc)
        logger.exception("TTS synthesis worker failed")
    finally:
        stats.synthesis_finished_at = _now()
        _put_chunk_with_backpressure(loop, queue, None, cancel_event)


def _log_stream_metrics(stats: _SynthesisStats) -> None:
    finished_at = stats.synthesis_finished_at or _now()
    first_byte_latency_ms: float | None = None
    if stats.first_chunk_emitted_at is not None:
        first_byte_latency_ms = (stats.first_chunk_emitted_at - stats.request_started_at) * 1000

    sr = model.sr if model is not None else 24000
    audio_duration_s = stats.emitted_samples / float(sr)
    total_latency_ms = (finished_at - stats.request_started_at) * 1000

    logger.info(
        "tts_stream_complete %s",
        {
            "request_id": stats.request_id,
            "backend": stats.backend,
            "chars": stats.chars,
            "generated_tokens": stats.generated_tokens,
            "emitted_chunks": stats.emitted_chunks,
            "emitted_bytes": stats.emitted_bytes,
            "audio_duration_s": round(audio_duration_s, 3),
            "first_byte_latency_ms": round(first_byte_latency_ms, 1)
            if first_byte_latency_ms is not None
            else None,
            "total_latency_ms": round(total_latency_ms, 1),
            "cancelled": stats.cancelled,
            "error": stats.error,
        },
    )


async def _stream_pcm_response(
    req: SpeechRequest,
    request: Request,
    *,
    request_id: str,
    audio_prompt_path: str | None,
):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=PCM_STREAM_QUEUE_MAXSIZE)
    cancel_event = threading.Event()
    stats = _SynthesisStats(
        request_id=request_id,
        request_started_at=_now(),
        chars=len(req.input),
    )

    worker = threading.Thread(
        target=_run_synthesis_worker,
        kwargs={
            "loop": loop,
            "queue": queue,
            "req": req,
            "audio_prompt_path": audio_prompt_path,
            "cancel_event": cancel_event,
            "stats": stats,
        },
        daemon=True,
        name=f"tts-{request_id[:8]}",
    )
    worker.start()

    try:
        while True:
            if await request.is_disconnected():
                stats.cancelled = True
                cancel_event.set()
                logger.info(
                    "tts_stream_client_disconnected %s",
                    {"request_id": request_id, "chars": len(req.input)},
                )
                break

            try:
                chunk = await asyncio.wait_for(
                    queue.get(),
                    timeout=PCM_STREAM_QUEUE_POLL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                if not worker.is_alive() and queue.empty():
                    break
                continue

            if chunk is None:
                break

            if stats.first_chunk_emitted_at is None:
                stats.first_chunk_emitted_at = _now()
                logger.info(
                    "tts_stream_first_chunk %s",
                    {
                        "request_id": request_id,
                        "chars": len(req.input),
                        "first_byte_latency_ms": round(
                            (stats.first_chunk_emitted_at - stats.request_started_at) * 1000,
                            1,
                        ),
                    },
                )

            yield chunk
    finally:
        cancel_event.set()
        await asyncio.to_thread(worker.join, 0.5)
        _log_stream_metrics(stats)


@app.on_event("startup")
async def load_model():
    global model
    logger.info("Loading Chatterbox-Turbo on CUDA...")
    t0 = _now()
    ckpt_dir = _resolve_turbo_checkpoint()
    logger.info("Using checkpoint: %s", ckpt_dir)
    model = ChatterboxTurboTTS.from_local(ckpt_dir, device="cuda")

    # Patch: norm_loudness returns float64 (via pyloudnorm scalar math),
    # but s3tokenizer expects float32. Cast output back to float32.
    _orig_norm = model.norm_loudness

    def _norm_f32(wav, sr, target_lufs=-27):
        out = _orig_norm(wav, sr, target_lufs)
        return out.astype(np.float32) if hasattr(out, "astype") else out

    model.norm_loudness = _norm_f32

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Chatterbox loaded in %.1fs (SR=%s)", _now() - t0, model.sr)


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
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/x-m4a",
        "audio/m4a",
        "audio/flac",
        "audio/x-flac",
    }
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            400,
            f"Unsupported type: {file.content_type}. Use WAV, MP3, M4A, or FLAC.",
        )

    contents = await file.read()
    max_bytes = 10 * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            400,
            f"File too large ({len(contents) / 1024 / 1024:.1f}MB). Max 10MB.",
        )

    ref_id = uuid.uuid4().hex[:12]
    suffix = FSPath(file.filename).suffix if file.filename else ".wav"
    ref_path = REFERENCE_DIR / f"{ref_id}{suffix}"
    ref_path.write_bytes(contents)

    try:
        duration = librosa.get_duration(path=str(ref_path))
    except Exception as exc:
        ref_path.unlink(missing_ok=True)
        raise HTTPException(400, "Cannot read audio file. Ensure it is a valid audio format.") from exc

    references[ref_id] = {
        "path": str(ref_path),
        "filename": file.filename or "reference",
        "duration": round(duration, 1),
    }
    logger.info("Reference uploaded: %s (%.1fs) -> %s", file.filename, duration, ref_id)
    return {"id": ref_id, "filename": file.filename, "duration_seconds": round(duration, 1)}


@app.delete("/v1/voice/reference/{ref_id}")
async def delete_reference(ref_id: str):
    """Delete a previously uploaded reference audio."""
    ref = references.pop(ref_id, None)
    if ref:
        path = ref.get("path")
        if isinstance(path, str):
            FSPath(path).unlink(missing_ok=True)
        logger.info("Reference deleted: %s", ref_id)
    return {"ok": True}


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, request: Request):
    if model is None:
        return Response(content="Model not loaded", status_code=503)

    audio_prompt_path: str | None = None
    if req.reference_id:
        ref = references.get(req.reference_id)
        if not ref:
            raise HTTPException(404, "Reference audio not found. Please re-upload.")
        path = ref.get("path")
        if isinstance(path, str):
            audio_prompt_path = path

    request_id = uuid.uuid4().hex[:12]
    response_format = (req.response_format or "wav").lower()

    logger.info(
        "tts_request_start %s",
        {
            "request_id": request_id,
            "chars": len(req.input),
            "response_format": response_format,
            "has_reference": bool(audio_prompt_path),
        },
    )

    if response_format == "pcm":
        stream = _stream_pcm_response(
            req,
            request,
            request_id=request_id,
            audio_prompt_path=audio_prompt_path,
        )
        return StreamingResponse(
            stream,
            media_type="audio/pcm",
            headers={"X-Request-Id": request_id},
        )

    # Backward-compatible non-streaming WAV path.
    t0 = _now()
    with MODEL_LOCK:
        wav = model.generate(
            req.input,
            audio_prompt_path=audio_prompt_path,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
            temperature=req.temperature,
        )

    if wav.dim() > 1:
        wav = wav.squeeze(0)
    samples = wav.detach().cpu().numpy()
    pcm = _float_to_pcm16(samples)
    pcm_bytes = pcm.tobytes()

    duration_s = len(samples) / model.sr
    total_ms = (_now() - t0) * 1000
    logger.info(
        "tts_request_complete %s",
        {
            "request_id": request_id,
            "chars": len(req.input),
            "response_format": "wav",
            "audio_duration_s": round(duration_s, 3),
            "first_byte_latency_ms": round(total_ms, 1),
            "total_latency_ms": round(total_ms, 1),
        },
    )

    wav_bytes = _pcm_bytes_to_wav_bytes(pcm_bytes, model.sr)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Generation-Time": f"{total_ms / 1000:.2f}", "X-Request-Id": request_id},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
