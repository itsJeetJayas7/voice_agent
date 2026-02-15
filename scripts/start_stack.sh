#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_DIR="${ROOT_DIR}/voice-agent"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

if command -v supervisorctl >/dev/null 2>&1 && supervisorctl status voice_stack:voice_agent >/dev/null 2>&1; then
  supervisorctl start voice_stack:* >/dev/null || true
  echo "Supervisor-managed voice stack detected."
  supervisorctl status voice_stack:* || true
  exit 0
fi

require_file() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

start_if_missing() {
  local name="$1"
  local pattern="$2"
  local command="$3"
  local log_file="$4"

  if pgrep -f "${pattern}" >/dev/null 2>&1; then
    echo "[skip] ${name} already running"
    return
  fi

  echo "[start] ${name}"
  setsid -f bash -lc "${command} >> '${log_file}' 2>&1"
  sleep 1

  if pgrep -f "${pattern}" >/dev/null 2>&1; then
    echo "[ok] ${name}"
  else
    echo "[fail] ${name} did not stay up. Check ${log_file}" >&2
    exit 1
  fi
}

require_file "${VOICE_DIR}/.venv/bin/activate"
require_file "${ROOT_DIR}/start_vllm.sh"
require_file "${ROOT_DIR}/scripts/run_voice_agent.sh"

start_if_missing \
  "LiveKit" \
  "livekit-server --config ${VOICE_DIR}/infra/livekit/livekit.yaml --dev" \
  "livekit-server --config '${VOICE_DIR}/infra/livekit/livekit.yaml' --dev" \
  "${LOG_DIR}/livekit.log"

start_if_missing \
  "Voxtral vLLM STT" \
  "vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602" \
  "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/start_vllm.sh'" \
  "${LOG_DIR}/vllm.log"

start_if_missing \
  "Chatterbox TTS server" \
  "python -m voice_agent.tts_server" \
  "cd '${VOICE_DIR}' && source .venv/bin/activate && python -m voice_agent.tts_server" \
  "${LOG_DIR}/tts_server.log"

start_if_missing \
  "Token server" \
  "uvicorn voice_agent.token_server.main:app --host 0.0.0.0 --port 8081" \
  "cd '${VOICE_DIR}' && source .venv/bin/activate && uvicorn voice_agent.token_server.main:app --host 0.0.0.0 --port 8081" \
  "${LOG_DIR}/token_server.log"

start_if_missing \
  "Voice agent" \
  "python -u -m voice_agent.main" \
  "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/run_voice_agent.sh'" \
  "${LOG_DIR}/agent.log"

start_if_missing \
  "Web UI" \
  "http.server 5173 -d ${VOICE_DIR}/web" \
  "python3 -m http.server 5173 -d '${VOICE_DIR}/web'" \
  "${LOG_DIR}/web.log"

echo

echo "Stack is up."
echo "Web UI: http://localhost:5173"
echo "Health: http://localhost:8081/health"
echo "Logs: ${LOG_DIR}"
