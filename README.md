# Voice Agent VPS Setup

This repository contains the full LiveKit + Pipecat voice-agent stack used on this VPS:

- LiveKit RTC server
- Voxtral realtime STT (`vLLM`)
- Chatterbox Turbo TTS server
- Token server (`FastAPI`)
- Voice agent runtime (`Pipecat`)
- Web UI (`voice-agent/web`)

## Repository Layout

- `voice-agent/` — main application code
- `start_vllm.sh` — starts Voxtral realtime STT server
- `scripts/start_stack.sh` — starts the complete stack
- `scripts/stop_stack.sh` — stops the complete stack
- `RUNBOOK.md` — Voxtral-specific operational runbook

## Prerequisites

- Ubuntu/Linux with NVIDIA GPU + CUDA drivers
- `python3.11+`
- `ffmpeg`
- `livekit-server` binary installed and available in `PATH`
- Cerebras API key

## Fresh Machine Setup

```bash
git clone https://github.com/itsJeetJayas7/voice_agent.git
cd voice_agent

# Voxtral runtime env
python3 -m venv voxtral-env
source voxtral-env/bin/activate
pip install -U pip
pip install "vllm>=0.10.0" "numpy<2"
deactivate

# Voice agent env
python3 -m venv voice-agent/.venv
source voice-agent/.venv/bin/activate
pip install -U pip
pip install -e "./voice-agent[dev]"
cp voice-agent/.env.example voice-agent/.env
# edit voice-agent/.env and set at minimum CEREBRAS_API_KEY
```

## Start Everything

```bash
./scripts/start_stack.sh
```

Open:

- Voice UI: `http://localhost:5173`
- Token health: `http://localhost:8081/health`

Use room `voice-room` (default) unless you also change `AGENT_ROOM` in `voice-agent/.env`.

## Stop Everything

```bash
./scripts/stop_stack.sh
```

## Quick Health Checks

```bash
ss -ltnp | rg ':7880|:8000|:8001|:8081|:5173'
curl -s http://localhost:7880
curl -s http://localhost:8081/health
curl -s http://localhost:8001/health
```

## Logs

Runtime logs are written to `logs/` by `scripts/start_stack.sh`:

- `logs/livekit.log`
- `logs/vllm.log`
- `logs/tts_server.log`
- `logs/token_server.log`
- `logs/agent.log`
- `logs/web.log`

Tail agent logs:

```bash
tail -f logs/agent.log
```

## Notes

- Secrets are intentionally not committed (`.env`, API keys, tokens).
- Runtime artifacts/logs are intentionally ignored by git.
- `voice-agent/README.md` contains deeper architecture details for the app itself.
