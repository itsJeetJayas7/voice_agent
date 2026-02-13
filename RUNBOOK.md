# Voxtral Realtime STT — Runbook

## Start Server

```bash
bash /root/start_vllm.sh 2>&1 | tee /root/vllm_server.log
```

Wait for `Application startup complete` and the line:
```
Route: /v1/realtime, Endpoint: realtime_endpoint
```

## Start UI

```bash
source /root/voxtral-env/bin/activate
python /root/gradio_realtime_stt.py --host localhost --port 8000
```

Open `http://localhost:7860` in a browser. Click **Start**, speak, click **Stop**.

## Health Check

```bash
curl -s http://localhost:8000/health
```

Should return `200 OK`.

## Run Validation

```bash
source /root/voxtral-env/bin/activate
python /root/test_realtime_validation.py --cycles 3
```

Expected: all cycles PASS, TTFT < 10s, word overlap ≥ 40%.

## Stop Server

```bash
pkill -f "vllm serve"
```

## Stop UI

```bash
pkill -f "gradio_realtime_stt.py"
```

## Cleanup / Reset

```bash
# Kill all related processes
pkill -f "vllm serve" 2>/dev/null
pkill -f "gradio_realtime_stt" 2>/dev/null
# Clear old logs
rm -f /root/vllm_server.log /root/gradio.log
```

## Key Configuration

| Setting | Value | File |
|---|---|---|
| Launch mode | PIECEWISE cudagraphs | `start_vllm.sh` |
| Max model len | 32768 (~45 min sessions) | `start_vllm.sh` |
| Transcription delay | 480 ms | `tekken.json` (model default) |
| Temperature | 0.0 | Server-side (hardcoded in vLLM) |
| Audio format | PCM16, 16 kHz, mono | Client-side |

## Known Limits

1. **Max session length** — `--max-model-len 32768` limits continuous transcription to ~45 minutes. Increase to 65536 or 131072 for longer sessions (requires more VRAM for RoPE pre-allocation).

2. **Single user** — The current setup runs one vLLM engine instance. Multiple concurrent users each open a websocket session; the engine can handle a few, but throughput degrades. For multi-user, use `--max-num-batched-tokens` tuning or multiple replicas.

3. **Accuracy gap vs hosted demo** — The hosted Mistral demo may use ensemble decoding, VAD pre-processing, or language-detection hinting not available in the open-source model. Local transcription at 480 ms delay matches published benchmarks for Voxtral Realtime but may trail the hosted experience for noisy/accented audio.

4. **Cudagraph warmup** — PIECEWISE mode has a ~10–30s warmup phase on first request as cudagraphs are captured. Subsequent requests are faster.

5. **No VAD** — The model processes all audio continuously. Long silences consume KV cache tokens. A VAD (voice activity detection) front-end would improve efficiency but is not part of the official vLLM pipeline.

6. **Gradio mic chunk size** — Gradio's streaming audio chunk size depends on the browser. Very small chunks increase websocket overhead; very large chunks increase latency. The current setup uses Gradio defaults (~0.5s chunks).
