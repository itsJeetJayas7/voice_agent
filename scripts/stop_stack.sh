#!/usr/bin/env bash
set -euo pipefail

stop_pattern() {
  local name="$1"
  local pattern="$2"

  if pgrep -f "${pattern}" >/dev/null 2>&1; then
    pkill -f "${pattern}"
    echo "[stop] ${name}"
  else
    echo "[skip] ${name} not running"
  fi
}

stop_pattern "Voice agent" "python -u -m voice_agent.main"
stop_pattern "Token server" "uvicorn voice_agent.token_server.main:app --host 0.0.0.0 --port 8081"
stop_pattern "Chatterbox TTS server" "python -m voice_agent.tts_server"
stop_pattern "Web UI" "http.server 5173 -d .*/voice-agent/web"
stop_pattern "Voxtral vLLM STT" "vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602"
stop_pattern "LiveKit" "livekit-server --config .*/voice-agent/infra/livekit/livekit.yaml --dev"
