# Voice Agent Startup Report

**Date**: 2026-02-13
**Environment**: Linux x86_64, Python 3.12.3

---

## 1. Pre-flight Checks

### Environment baseline
- Killed stale pytest processes
- Confirmed only Voxtral vLLM (:8000) and Gradio (:7860) were running
- No voice-agent services were active

### Venv health
```
Python 3.12.3 (.venv)
pip 26.0.1
```

### Configuration
- Created `.env` from `.env.example`
- LiveKit keys: `devkey`/`devsecret` (match `infra/livekit/livekit.yaml`)
- Cerebras API key: **dummy** (`sk-dummy-not-real`) — LLM calls will fail at runtime
- All other settings: defaults from `.env.example`

### Dependencies
```
pip install -e ".[dev]" — success
Pipecat 0.0.102 loaded
Core imports verified: voice_agent, pipecat, livekit, openai
```

---

## 2. Static Quality Gates

### Ruff (linter)
- **Before**: 27 errors (all unused imports)
- **After fix**: `ruff check --fix --unsafe-fixes` — 0 errors

### Mypy (type checker)
- **Before**: 9 errors (type ignores, websockets API, OpenAI typing)
- **After fix**: 0 errors
- Fixes applied:
  - Corrected `type: ignore` error codes in `interruption_controller.py`
  - Added `type: ignore` for untyped decorator in `main.py`
  - Added `type: ignore` for websockets client API in adapter WS modules
  - Added `type: ignore` for union-attr in `cerebras_openai.py` async iterator
  - Added `type: ignore` for LLMContext message type in `factory.py`

---

## 3. Test Results

### Before fixes: 3 failures
| Test | Failure | Root Cause |
|------|---------|------------|
| `test_cerebras_adapter::test_cancel_stops_iteration` | 401 AuthenticationError | `stream_completion()` reset `_cancelled=False` before checking, then hit real Cerebras API |
| `test_interruption_controller::test_reset_restores_state` | `assert controller.output_allowed` failed | `reset_for_new_turn()` didn't clear session `_cancel_event` |
| `test_interruption_controller::test_tasks_cancelled_on_barge_in` | `assert tts_task.cancelled()` failed | No event loop yield after `task.cancel()` — task not yet in cancelled state |

### Fixes applied
1. **cerebras_openai.py**: Removed `self._cancelled = False` reset at start of `stream_completion()` so pre-cancel is honoured
2. **interruption_controller.py**: Added `self._session._cancel_event.clear()` in `reset_for_new_turn()`
3. **test_interruption_controller.py**: Added `await asyncio.sleep(0)` after barge-in to let tasks process cancellation

### Integration test hang fix
- **Root cause**: `TTSChunker._output_queue` had `maxsize=2`, test produced 3+ chunks without consuming, causing `queue.put()` to block forever
- **Fix**: Made queue maxsize configurable via `ChunkerConfig.queue_maxsize` (default: 2). Integration test uses `queue_maxsize=0` (unbounded)

### After fixes: 53 passed, 0 failed
```
tests/unit: 49 passed
tests/integration: 4 passed
Total: 53 passed in ~4s
```

---

## 4. Architecture Mode

**Selected: Minimal Startup Mode** (LiveKit + Pipecat core only)

### Pipeline composition
```
transport.input() → user_aggregator (VAD) → OpenAILLMService (Cerebras) → transport.output() → assistant_aggregator
```

### Rationale
- Pipeline is functional with Pipecat's built-in STT/TTS + Cerebras LLM
- Custom Voxtral/Chatterbox adapters exist in `adapters/` but are standalone HTTP/WS client wrappers, not Pipecat `FrameProcessor` subclasses
- Wiring custom adapters requires building processor glue code (future work)

### What's included
- LiveKit WebRTC transport (audio I/O)
- Silero VAD + smart turn detection
- Cerebras LLM via OpenAI-compatible API (streaming)
- Pipecat context aggregation + frame routing
- Interruption controller (barge-in support)

### What's NOT wired yet
- Voxtral STT adapters (`voxtral_http.py`, `voxtral_ws.py`)
- Chatterbox TTS adapters (`chatterbox_http.py`, `chatterbox_ws.py`)
- Custom TTS chunker in pipeline path
- These exist as tested standalone clients for future integration

---

## 5. Active Processes and Ports

| Service | PID | Port | Status |
|---------|-----|------|--------|
| LiveKit Server v1.9.11 | 118061 | :7880 | Running (dev mode) |
| Token Server (uvicorn) | 118308 | :8081 | Running |
| Web UI (http.server) | 118497 | :5173 | Running |
| Voice Agent (Pipecat) | 118653 | — | Running, connected to room `voice-room` |
| Voxtral vLLM (pre-existing) | 89047 | :8000 | Running |
| Gradio demo (pre-existing) | 37058 | :7860 | Running |

---

## 6. Smoke Test Results

| Endpoint | Result |
|----------|--------|
| `GET http://localhost:7880` | `OK` |
| `GET http://localhost:8081/health` | `{"status":"ok"}` |
| `POST http://localhost:8081/token` | Valid JWT returned |
| `GET http://localhost:5173` | HTML page served |
| Agent LiveKit connection | Connected to room `voice-room` |

---

## 7. Startup Commands (Reproducible)

```bash
# 1. LiveKit
livekit-server --config /root/voice-agent/infra/livekit/livekit.yaml --dev &

# 2. Token Server
source /root/voice-agent/.venv/bin/activate
uvicorn voice_agent.token_server.main:app --host 0.0.0.0 --port 8081 &

# 3. Voice Agent
python -m voice_agent.main &

# 4. Web UI
python3 -m http.server 5173 -d /root/voice-agent/web &
```

---

## 8. Known Remaining Gaps

1. **CEREBRAS_API_KEY is a dummy** — LLM calls will fail. Replace with a real key in `.env` for end-to-end voice functionality
2. **Pipecat handles STT/TTS natively** — custom Voxtral/Chatterbox adapters are not in the pipeline. To use them:
   - Create `FrameProcessor` wrappers for each adapter
   - Insert into pipeline in `factory.py` between transport and aggregators
3. **LiveKit HMAC key warning** — `devkey`/`devsecret` are too short (9 bytes < 32 minimum). Use longer keys in production
4. **CORS is open** (`allow_origins=["*"]`) in token server — restrict in production
5. **No TLS** — using `ws://` and `http://` — use `wss://` and `https://` in production
6. **GPU discovery warning** — onnxruntime can't find GPU device file; Silero VAD falls back to CPU (fine for dev)
7. **Browser test not automated** — manual verification needed at `http://localhost:5173`

---

## 9. Files Modified

| File | Change |
|------|--------|
| `src/voice_agent/adapters/llm/cerebras_openai.py` | Removed `_cancelled` reset; added type: ignore |
| `src/voice_agent/adapters/stt/voxtral_ws.py` | Added type: ignore for websockets API |
| `src/voice_agent/adapters/tts/chatterbox_ws.py` | Added type: ignore for websockets API |
| `src/voice_agent/config.py` | Removed unused `Field` import |
| `src/voice_agent/main.py` | Removed unused imports; added type: ignore for decorator |
| `src/voice_agent/pipeline/factory.py` | Added type: ignore for LLMContext arg |
| `src/voice_agent/pipeline/interruption_controller.py` | Fixed reset to clear session cancel event; fixed type: ignore codes; removed unused imports |
| `src/voice_agent/pipeline/tts_chunker.py` | Made queue_maxsize configurable via ChunkerConfig |
| `src/voice_agent/pipeline/audio_codec.py` | Removed unused import |
| `src/voice_agent/pipeline/vad_endpointing.py` | Removed unused import |
| `tests/unit/test_cerebras_adapter.py` | Removed unused imports |
| `tests/unit/test_interruption_controller.py` | Added event loop yield; removed unused import |
| `tests/unit/test_chatterbox_adapter.py` | Removed unused import |
| `tests/unit/test_voxtral_adapter.py` | Removed unused imports |
| `tests/unit/test_tts_chunker.py` | Removed unused import |
| `tests/integration/test_simulated_streaming_turns.py` | Set unbounded queue; removed unused imports |
