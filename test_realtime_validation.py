#!/usr/bin/env python3
"""
Validation harness for Voxtral Realtime STT.

Tests:
  1. Single-session transcription of known audio (mary_had_lamb).
  2. Transcript quality check (key phrases must appear).
  3. TTFT measurement (time-to-first-token).
  4. Multi-cycle stability (N connect/stream/disconnect cycles).
  5. Server health check after each cycle.

Usage:
  source /root/voxtral-env/bin/activate
  python test_realtime_validation.py [--host localhost] [--port 8000] [--cycles 3]
"""

import argparse
import asyncio
import base64
import json
import sys
import time

import numpy as np
import requests
import websockets

# ---------------------------------------------------------------------------
# Reference: "Mary had a little lamb" asset from vLLM
# Known transcript (approximate):
#   "Mary had a little lamb, its fleece was white as snow,
#    and everywhere that Mary went, the lamb was sure to go."
# The asset also has a preamble about Edison's phonograph.
# ---------------------------------------------------------------------------

REFERENCE_PHRASES = [
    "Mary had a little lamb",
    "lamb was sure to go",
]

# At least this fraction of reference words should appear
MIN_WORD_OVERLAP = 0.40


def load_test_audio():
    """Load the mary_had_lamb audio asset and return (pcm16_bytes, duration_s)."""
    import librosa
    from vllm.assets.audio import AudioAsset

    path = str(AudioAsset("mary_had_lamb").get_local_path())
    audio, _ = librosa.load(path, sr=16000, mono=True)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes(), len(audio) / 16000


def word_overlap(transcript: str, reference_phrases: list[str]) -> float:
    """Return fraction of reference words found in transcript (case-insensitive)."""
    t_words = set(transcript.lower().split())
    ref_words = set()
    for phrase in reference_phrases:
        ref_words.update(phrase.lower().split())
    if not ref_words:
        return 0.0
    return len(t_words & ref_words) / len(ref_words)


async def run_single_session(
    host: str, port: int, model: str, audio_bytes: bytes, chunk_size: int = 4096
) -> dict:
    """
    Run one realtime WS session: stream audio → collect transcript → close.

    Returns dict with keys: transcript, ttft_s, duration_s, error
    """
    uri = f"ws://{host}:{port}/v1/realtime"
    result = {"transcript": "", "ttft_s": None, "duration_s": 0, "error": None}

    t_start = time.monotonic()
    first_token_time = None

    try:
        async with websockets.connect(uri) as ws:
            # session.created
            resp = json.loads(await ws.recv())
            session_id = resp.get("id", "unknown")
            print(f"  Session: {session_id}")

            # session.update
            await ws.send(json.dumps({"type": "session.update", "model": model}))
            # initial commit → starts generation
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            # Send audio chunks
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i : i + chunk_size]
                b64 = base64.b64encode(chunk).decode("utf-8")
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.append", "audio": b64})
                )

            # final commit
            await ws.send(
                json.dumps({"type": "input_audio_buffer.commit", "final": True})
            )

            # Collect transcription
            transcript = ""
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
                data = json.loads(msg)
                t = data.get("type", "")
                if t == "transcription.delta":
                    delta = data.get("delta", "")
                    if delta and first_token_time is None:
                        first_token_time = time.monotonic()
                    transcript += delta
                elif t == "transcription.done":
                    break
                elif t == "error":
                    result["error"] = data.get("error", str(data))
                    break

            result["transcript"] = transcript.strip()
            t_end = time.monotonic()
            result["duration_s"] = t_end - t_start
            if first_token_time is not None:
                result["ttft_s"] = first_token_time - t_start

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def check_health(host: str, port: int) -> bool:
    """Hit /health and return True if server is healthy."""
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


async def main(args):
    print("=" * 60)
    print("Voxtral Realtime STT — Validation Harness")
    print("=" * 60)

    # Pre-flight health check
    if not check_health(args.host, args.port):
        print("FAIL: Server not healthy at startup. Is vLLM running?")
        sys.exit(1)
    print(f"Server healthy at http://{args.host}:{args.port}")

    audio_bytes, audio_duration = load_test_audio()
    print(f"Test audio: {audio_duration:.1f}s (mary_had_lamb)")
    print()

    results = []
    for cycle in range(1, args.cycles + 1):
        print(f"--- Cycle {cycle}/{args.cycles} ---")
        r = await run_single_session(
            args.host, args.port, args.model, audio_bytes
        )
        results.append(r)

        overlap = word_overlap(r["transcript"], REFERENCE_PHRASES)
        ttft_str = f'{r["ttft_s"]:.2f}s' if r["ttft_s"] is not None else "N/A"
        print(f"  Transcript: {r['transcript'][:120]}…")
        print(f"  TTFT: {ttft_str}")
        print(f"  Duration: {r['duration_s']:.2f}s")
        print(f"  Word overlap: {overlap:.0%}")
        if r["error"]:
            print(f"  ERROR: {r['error']}")

        # Health check after each cycle
        healthy = check_health(args.host, args.port)
        print(f"  Health: {'OK' if healthy else 'FAIL'}")
        if not healthy:
            print("FAIL: Server crashed after cycle. Aborting.")
            sys.exit(1)
        print()

    # ---- Summary ----
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Cycle':<8} {'TTFT':>8} {'Duration':>10} {'Overlap':>10} {'Error'}")
    print("-" * 60)
    all_pass = True
    for i, r in enumerate(results, 1):
        overlap = word_overlap(r["transcript"], REFERENCE_PHRASES)
        ttft_str = f'{r["ttft_s"]:.2f}s' if r["ttft_s"] is not None else "N/A"
        err_str = r["error"] or "—"
        print(f"{i:<8} {ttft_str:>8} {r['duration_s']:>9.2f}s {overlap:>9.0%} {err_str}")
        if r["error"] or overlap < MIN_WORD_OVERLAP:
            all_pass = False

    print()
    if all_pass:
        print("RESULT: PASS ✅")
    else:
        print("RESULT: FAIL ❌")
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxtral Realtime Validation")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="mistralai/Voxtral-Mini-4B-Realtime-2602")
    parser.add_argument("--cycles", type=int, default=3, help="Number of start/stop cycles")
    args = parser.parse_args()
    rc = asyncio.run(main(args))
    sys.exit(rc)
