# RUNBOOK — Voice Agent Operations

## Startup Checklist

1. **Environment** — Verify `.env` is populated with real keys (never commit secrets)
2. **LiveKit** — Either local Docker or LiveKit Cloud is reachable
3. **STT Service** — Voxtral is running and reachable at `VOXTRAL_HOST:VOXTRAL_PORT`
4. **TTS Service** — Chatterbox is running and reachable at `CHATTERBOX_HOST:CHATTERBOX_PORT`
5. **LLM** — Cerebras API key is valid and quota is available
6. **Token Server** — Running on `TOKEN_SERVER_PORT` (default 8081)
7. **Agent** — Running and joined the LiveKit room

## Troubleshooting

### Token server returns 500
- Check `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` match LiveKit config
- Verify `LIVEKIT_URL` is correct (include `ws://` or `wss://`)

### Agent doesn't join room
- Verify LiveKit is running: `curl http://localhost:7880`
- Check token generation succeeds: `curl -X POST localhost:8081/token -H 'Content-Type: application/json' -d '{"room":"test","identity":"agent","participant_type":"agent"}'`

### No audio from agent
- Verify Cerebras API key is valid and model is available
- Check agent logs for LLM errors (timeout, 429, 5xx)
- Verify TTS service is reachable

### High latency
- Tune `CHUNK_MIN_CHARS` lower (e.g., 16) for faster first TTS
- Reduce `CHUNK_MAX_WAIT_MS` (e.g., 200ms)
- Lower `VAD_START_SECS` for faster barge-in detection
- Check network latency to Cerebras API
- Verify TTS service is not overloaded

### Barge-in not working
- Verify `enable_interruptions` is true in transport config
- Check VAD sensitivity: lower `VAD_CONFIDENCE` (e.g., 0.5)
- Reduce `VAD_START_SECS` for faster speech detection

### STT/TTS service crashes
- Check GPU VRAM usage: `nvidia-smi`
- Reduce `MAX_CONCURRENT_SESSIONS` to limit resource pressure
- Agent will log errors and attempt clean recovery per-turn

## Tuning Guide

### Latency Optimization
| Metric | Target | Tuning |
|--------|--------|--------|
| LLM TTFT | < 200ms | Use Cerebras (fast inference) |
| TTS first audio | < 200ms | Lower `CHUNK_MIN_CHARS` |
| Barge-in cutoff | < 50ms | Lower `VAD_START_SECS` |
| E2E first audio | < 600ms | All of the above |

### Quality vs. Speed Tradeoffs
- **Shorter chunks** = lower latency but potentially choppy speech
- **Longer endpointing silence** = fewer false endpoints but slower response
- **Lower VAD confidence** = faster barge-in but more false triggers

## Metrics

Structured logs include latency metrics per turn:
- `llm_ttft_ms` — LLM request start → first token
- `tts_first_audio_latency_ms` — First text chunk → first audio frame
- `barge_in_cutoff_ms` — Speech start → playback halted
- `e2e_first_audio_ms` — User turn end → first agent audio
- `endpointer_latency_ms` — Last speech → turn end decision

To aggregate: filter JSON logs for `"event": "Turn metrics"`.

## Security Notes

- **Secrets**: Never log API keys or auth headers (enforced in logging module)
- **Transcripts**: Set `LOG_TRANSCRIPTS=false` (default) in production to avoid PII logging
- **CORS**: Restrict `allow_origins` in `token_server/main.py` for production
- **TLS**: Use `wss://` for LiveKit and `https://` for all service endpoints in production
- **Input validation**: Room/identity names are restricted to `[a-zA-Z0-9_-]`
- **Token TTL**: Default 1 hour; reduce for higher security environments

## PII & Data Minimization

- Audio is processed in-memory only; no persistent storage by default
- Transcript logging is opt-in (`LOG_TRANSCRIPTS=true`)
- Session cleanup is deterministic: all buffers and queues are cleared on disconnect
- No user audio is forwarded to external services beyond STT (local)

## Failure Handling

| Failure | Behaviour |
|---------|-----------|
| STT timeout | Per-request timeout + retry (where safe) |
| TTS timeout | Per-request timeout; emit silence frame |
| LLM 429 | Exponential backoff with jitter, max 3 retries |
| LLM 5xx | Exponential backoff, fail fast after retries |
| LLM non-retryable | Fail fast, log error |
| LiveKit disconnect | Session cleanup triggers; reconnect via client |
| Adapter crash mid-turn | Cancel turn, recover cleanly for next turn |

## Manual Smoke Test

1. Start all services (LiveKit, token server, agent)
2. Open `http://localhost:5173` in browser
3. Enter room name and click Connect
4. Speak a sentence — verify partial transcripts during speech
5. Confirm agent starts speaking before full response completes
6. Interrupt agent mid-sentence — verify audio stops immediately
7. Open two browser tabs with different identities — verify isolation
8. Kill STT/TTS service — verify graceful error logging and recovery
