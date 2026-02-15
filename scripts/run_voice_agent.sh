#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_DIR="${ROOT_DIR}/voice-agent"

wait_http() {
  local name="$1"
  local url="$2"
  local timeout_s="${3:-180}"
  local started_at
  started_at="$(date +%s)"

  while true; do
    if curl -fsS -m 3 "${url}" >/dev/null 2>&1; then
      echo "[ready] ${name} (${url})"
      return
    fi
    if [ "$(($(date +%s) - started_at))" -ge "${timeout_s}" ]; then
      echo "[fail] ${name} did not become ready within ${timeout_s}s: ${url}" >&2
      exit 1
    fi
    sleep 2
  done
}

if [ ! -f "${VOICE_DIR}/.venv/bin/activate" ]; then
  echo "Missing ${VOICE_DIR}/.venv/bin/activate" >&2
  exit 1
fi

# Prevent agent boot before dependencies are actually serving requests.
wait_http "LiveKit" "http://localhost:7880/" 30
wait_http "Voxtral vLLM STT" "http://localhost:8000/health" 300
wait_http "Chatterbox TTS" "http://localhost:8001/health" 120
wait_http "Token server" "http://localhost:8081/health" 60

cd "${VOICE_DIR}"
source .venv/bin/activate
exec python -u -m voice_agent.main
