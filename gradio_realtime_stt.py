# SPDX-License-Identifier: Apache-2.0
# Based on vLLM's openai_realtime_microphone_client.py example
"""
Gradio demo for real-time speech transcription using the vLLM Realtime API.

Start the vLLM server first:

  bash /root/start_vllm.sh

Then run this script:

  python gradio_realtime_stt.py --host localhost --port 8000

Local-only: --share is removed; UI binds to 0.0.0.0 for LAN access.
"""

import argparse
import asyncio
import base64
import json
import logging
import queue
import threading
import time

import gradio as gr
import numpy as np
import websockets

# --------------- configuration ---------------
SAMPLE_RATE = 16_000
STOP_SENTINEL = "__STOP__"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voxtral-ui")

# --------------- global state ---------------
audio_queue: queue.Queue = queue.Queue()
transcription_text = ""
is_running = False
ws_url = ""
model = ""

# Session singleton: the one active websocket thread (or None)
_ws_thread: threading.Thread | None = None
_ws_thread_lock = threading.Lock()
# Session id for logging
_session_id: str = ""


# --------------- websocket handler ---------------
async def websocket_handler():
    """Connect to WebSocket and handle audio streaming + transcription."""
    global transcription_text, is_running, _session_id

    log.info("Connecting to %s …", ws_url)
    async with websockets.connect(ws_url) as ws:
        # 1) Wait for session.created
        resp = json.loads(await ws.recv())
        _session_id = resp.get("id", "unknown")
        log.info("Session created: %s", _session_id)

        # 2) Validate model
        await ws.send(json.dumps({"type": "session.update", "model": model}))

        # 3) Signal ready → first (non-final) commit starts generation
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        log.info("[%s] Generation started, ready to receive audio", _session_id)

        async def send_audio():
            """Read from audio_queue and forward to WS."""
            chunks_sent = 0
            while True:
                try:
                    item = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: audio_queue.get(timeout=0.2)
                    )
                except queue.Empty:
                    if not is_running:
                        # Stopped and queue drained → send final commit
                        break
                    continue

                if item is STOP_SENTINEL:
                    break

                await ws.send(
                    json.dumps(
                        {"type": "input_audio_buffer.append", "audio": item}
                    )
                )
                chunks_sent += 1
                if chunks_sent % 100 == 0:
                    log.info(
                        "[%s] Sent %d chunks (queue depth %d)",
                        _session_id,
                        chunks_sent,
                        audio_queue.qsize(),
                    )

            # Signal end-of-audio
            log.info(
                "[%s] Sending final commit after %d chunks", _session_id, chunks_sent
            )
            await ws.send(
                json.dumps({"type": "input_audio_buffer.commit", "final": True})
            )

        async def receive_transcription():
            """Read transcription deltas from WS."""
            global transcription_text
            async for message in ws:
                data = json.loads(message)
                msg_type = data.get("type", "")
                if msg_type == "transcription.delta":
                    delta = data.get("delta", "")
                    if delta:
                        transcription_text += delta
                elif msg_type == "transcription.done":
                    log.info("[%s] Transcription done", _session_id)
                    break
                elif msg_type == "error":
                    log.error("[%s] Server error: %s", _session_id, data)
                    break

        await asyncio.gather(send_audio(), receive_transcription())

    log.info("[%s] WebSocket closed cleanly", _session_id)


def _run_ws_loop():
    """Entry-point for the websocket background thread."""
    global is_running
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(websocket_handler())
    except Exception as e:
        log.error("WebSocket error: %s: %s", type(e).__name__, e)
    finally:
        is_running = False
        log.info("WebSocket thread exiting")


# --------------- Gradio callbacks ---------------
def start_recording():
    """Start the transcription service (singleton guard)."""
    global transcription_text, is_running, _ws_thread

    with _ws_thread_lock:
        if _ws_thread is not None and _ws_thread.is_alive():
            log.warning("Session already running — ignoring Start")
            return (
                gr.update(interactive=False),
                gr.update(interactive=True),
                transcription_text,
            )

        # Reset state
        transcription_text = ""
        _drain_queue()
        is_running = True

        log.info("Starting new recording session")
        _ws_thread = threading.Thread(target=_run_ws_loop, daemon=True)
        _ws_thread.start()

    return gr.update(interactive=False), gr.update(interactive=True), ""


def stop_recording():
    """Stop the transcription service gracefully."""
    global is_running, _ws_thread

    log.info("Stop requested")
    is_running = False
    # Push sentinel so send_audio() exits even if queue was empty
    audio_queue.put(STOP_SENTINEL)

    # Wait briefly for thread to finish cleanly
    with _ws_thread_lock:
        if _ws_thread is not None and _ws_thread.is_alive():
            _ws_thread.join(timeout=5.0)
            if _ws_thread.is_alive():
                log.warning("WebSocket thread did not exit in 5 s")
        _ws_thread = None

    return gr.update(interactive=True), gr.update(interactive=False), transcription_text


def process_audio(audio):
    """Process incoming audio from Gradio mic and queue for streaming."""
    global transcription_text

    if audio is None or not is_running:
        return transcription_text

    sample_rate, audio_data = audio

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Normalize to float32
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    elif audio_data.dtype == np.float64:
        audio_float = audio_data.astype(np.float32)
    else:
        audio_float = audio_data.astype(np.float32)

    # Resample to 16 kHz if needed (use proper sinc resampling)
    if sample_rate != SAMPLE_RATE:
        try:
            import soxr
            audio_float = soxr.resample(audio_float, sample_rate, SAMPLE_RATE)
        except ImportError:
            # Fallback: simple linear interp (not ideal but functional)
            num_samples = int(len(audio_float) * SAMPLE_RATE / sample_rate)
            audio_float = np.interp(
                np.linspace(0, len(audio_float) - 1, num_samples),
                np.arange(len(audio_float)),
                audio_float,
            )

    # Convert to PCM16 and base64 encode
    pcm16 = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
    b64_chunk = base64.b64encode(pcm16.tobytes()).decode("utf-8")
    audio_queue.put(b64_chunk)

    return transcription_text


def _drain_queue():
    """Empty the audio queue."""
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


# --------------- Gradio UI ---------------
with gr.Blocks(title="Voxtral Real-time Speech Transcription") as demo:
    gr.Markdown(
        """
    # Voxtral Mini 4B — Real-time Speech Transcription
    **Model:** `mistralai/Voxtral-Mini-4B-Realtime-2602` served via vLLM

    Click **Start** and speak into your microphone. Click **Stop** to end.
    """
    )

    with gr.Row():
        start_btn = gr.Button("Start", variant="primary", scale=1)
        stop_btn = gr.Button("Stop", variant="stop", interactive=False, scale=1)

    audio_input = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="numpy",
        label="Microphone Input",
    )
    transcription_output = gr.Textbox(
        label="Transcription",
        lines=10,
        placeholder="Transcription will appear here…",
    )

    start_btn.click(
        start_recording, outputs=[start_btn, stop_btn, transcription_output]
    )
    stop_btn.click(
        stop_recording, outputs=[start_btn, stop_btn, transcription_output]
    )
    audio_input.stream(
        process_audio, inputs=[audio_input], outputs=[transcription_output]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voxtral Realtime WebSocket Transcription with Gradio"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Voxtral-Mini-4B-Realtime-2602",
        help="Model being served by vLLM.",
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="vLLM server host"
    )
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument(
        "--gradio-port", type=int, default=7860, help="Gradio UI port"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create public Gradio link"
    )
    args = parser.parse_args()

    ws_url = f"ws://{args.host}:{args.port}/v1/realtime"
    model = args.model
    log.info("vLLM realtime endpoint: %s", ws_url)
    log.info("Model: %s", model)

    # Print clickable localhost link for user convenience
    print(f"Running on local URL:  http://localhost:{args.gradio_port}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.gradio_port,
        share=args.share,
    )
