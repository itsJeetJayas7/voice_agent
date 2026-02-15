# Voice Agent

Ultra-low-latency, interruptible voice agent built with **LiveKit** + **Pipecat** + **Voxtral STT** + **Chatterbox TTS** + **Cerebras LLM**.

## Architecture

```
┌──────────────────────────────┐
│ Web/Mobile Client            │
│ (LiveKit JS SDK)             │
│ - publish mic track          │
│ - play remote agent audio    │
└──────────────┬───────────────┘
               │ WebRTC (audio tracks + data)
               ▼
┌────────────────────────────────────────────────────────────┐
│ LiveKit Room                                               │
│ - user participant(s)                                      │
│ - agent participant (this service via Pipecat transport)   │
└──────────────┬─────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────┐
│ Voice Agent Service (Pipecat + Session Manager)            │
│                                                            │
│  LiveKitTransport.input()                                  │
│      → VAD + Endpointing                                   │
│      → Voxtral Realtime STT (WS /v1/realtime, partial/final)│
│      → LLM Context Aggregation                             │
│      → Cerebras LLM Adapter (stream tokens)                │
│      → Adaptive Speakable Chunker                          │
│      → Chatterbox TTS Adapter (stream/fallback chunk)      │
│      → LiveKitTransport.output()                           │
│                                                            │
│  Interruption Controller (speech-start barge-in)           │
│  - stop output immediately                                 │
│  - cancel TTS task → cancel LLM task → reset turn          │
└────────────────────────────────────────────────────────────┘
```

## Why This Design Achieves Low Latency

- **Fully streaming path** at every stage: STT partials → LLM tokens → incremental TTS
- **Early chunked TTS** starts speaking before the full LLM completion arrives
- **Local VAD** gives near-immediate speech-start detection for barge-in
- **True cancellation** avoids stale buffered audio and long interruption tails
- **Bounded queues + short audio frames** (10–20ms) reduce buffering delay
- **Session isolation** avoids global locks and contention under concurrency

## Assumptions

- Voxtral STT and Chatterbox Turbo TTS are **already running** as services; this repo implements client adapters only
- LiveKit can run locally with Docker **or** on LiveKit Cloud — both paths included
- One local GPU with enough VRAM runs STT/TTS; LLM is remote on Cerebras

## Quick Start

### 1. Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your actual keys
```

### 2. Start LiveKit

**Option A: Local Docker**
```bash
make livekit-up
```

**Option B: LiveKit Cloud**
```bash
# Set LIVEKIT_URL to your cloud ws URL and use cloud API key/secret in .env
```

### 3. Run Token Server

```bash
make token-server
# or: uvicorn src.voice_agent.token_server.main:app --host 0.0.0.0 --port 8081 --reload
```

### 4. Run Agent

```bash
make agent
# or: python -m src.voice_agent.main
```

### 5. Open Web Client

```bash
make web
# Open http://localhost:5173
```

### 6. Run Tests

```bash
make test
# or: pytest -q
```

## Project Structure

```
voice-agent/
├── pyproject.toml
├── Makefile
├── .env.example
├── infra/livekit/              # Local LiveKit Docker setup
│   ├── docker-compose.yml
│   └── livekit.yaml
├── web/                        # Minimal web client
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── src/voice_agent/
│   ├── config.py               # Pydantic-settings config
│   ├── logging.py              # Structured JSON logging
│   ├── metrics.py              # Latency metrics collector
│   ├── session.py              # Per-room Session object
│   ├── session_manager.py      # Concurrent session manager
│   ├── lifecycle.py            # Room lifecycle callbacks
│   ├── main.py                 # Agent entrypoint
│   ├── token_server/
│   │   ├── main.py             # FastAPI POST /token
│   │   └── schemas.py          # Request/response models
│   ├── pipeline/
│   │   ├── factory.py          # Pipeline composition
│   │   ├── audio_codec.py      # PCM16 normalize/resample
│   │   ├── vad_endpointing.py  # VAD + endpointing
│   │   ├── tts_chunker.py      # Adaptive speakable chunker
│   │   └── interruption_controller.py
│   └── adapters/
│       ├── stt/
│       │   ├── base.py         # STT protocol interface
│       │   ├── voxtral_http.py # HTTP transport
│       │   └── voxtral_ws.py   # WebSocket transport
│       ├── llm/
│       │   ├── base.py         # LLM protocol interface
│       │   └── cerebras_openai.py  # Cerebras streaming
│       └── tts/
│           ├── base.py         # TTS protocol interface
│           ├── chatterbox_http.py  # HTTP transport
│           └── chatterbox_ws.py    # WebSocket transport
└── tests/
    ├── unit/
    │   ├── test_tts_chunker.py
    │   ├── test_vad_endpointing.py
    │   ├── test_cerebras_adapter.py
    │   ├── test_voxtral_adapter.py
    │   ├── test_chatterbox_adapter.py
    │   └── test_interruption_controller.py
    └── integration/
        └── test_simulated_streaming_turns.py
```

## Configuration

All configuration is via environment variables (`.env` file). See `.env.example` for the complete list.

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `LIVEKIT_URL` | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | LiveKit API key |
| `LIVEKIT_API_SECRET` | LiveKit API secret |
| `CEREBRAS_API_KEY` | Cerebras inference API key |
| `CEREBRAS_MODEL` | LLM model name (default: `gpt-oss-120b`) |

### Performance Knobs

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIO_FRAME_MS` | 20 | Audio frame duration in ms |
| `CHUNK_MIN_CHARS` | 24 | Min chars before TTS chunk flush |
| `CHUNK_MAX_CHARS` | 120 | Max chars before forced flush |
| `CHUNK_MAX_WAIT_MS` | 350 | Max time before forced flush |
| `VAD_START_SECS` | 0.2 | Speech start confirmation window |
| `VAD_STOP_SECS` | 0.8 | Speech stop window |
| `ENDPOINT_MIN_SILENCE_MS` | 300 | Min silence with STT final |
| `ENDPOINT_MAX_SILENCE_MS` | 1500 | Max silence without STT final |
| `MAX_CONCURRENT_SESSIONS` | 10 | Session capacity cap |
| `AGENT_IDLE_TIMEOUT_SECS` | `none` | Idle timeout before pipeline is considered idle (`none` keeps agent always-on) |
| `AGENT_CANCEL_ON_IDLE_TIMEOUT` | `false` | Whether to cancel pipeline when idle timeout is hit |

### STT Realtime / Fallback Tuning

The active STT path uses a **Voxtral realtime websocket** connection (`/v1/realtime`), not the HTTP adapter.
Audio is streamed via `input_audio_buffer.append`, and turn boundaries are signaled by `input_audio_buffer.commit` with `final=true`.

By default, final commits are driven **only by VAD stop events** (local Silero VAD).
A time-based fallback commit mechanism exists but is **disabled by default** to prevent premature turn finalization.

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_FALLBACK_COMMIT_ENABLED` | `false` | Enable time-based fallback commits (only if VAD stops are unreliable) |
| `STT_FALLBACK_COMMIT_INTERVAL_S` | `5.0` | Min seconds between fallback commits |
| `STT_FALLBACK_MIN_VOICED_APPENDS` | `20` | Min voiced audio chunks before fallback can fire |
| `STT_VOICED_PEAK_THRESHOLD` | `80` | PCM16 peak amplitude to count a chunk as voiced |
| `STT_VAD_STOP_DEBOUNCE_MS` | `150` | Debounce window to suppress duplicate VAD-stop commits |

## License

MIT
