#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VOXTRAL_VENV="${VOXTRAL_VENV:-${SCRIPT_DIR}/voxtral-env}"

if [ ! -f "${VOXTRAL_VENV}/bin/activate" ]; then
  echo "Voxtral virtualenv not found at ${VOXTRAL_VENV}."
  echo "Create it with: python3 -m venv ${VOXTRAL_VENV}"
  exit 1
fi

# shellcheck source=/dev/null
source "${VOXTRAL_VENV}/bin/activate"

export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"

MODEL="${VOXTRAL_MODEL:-mistralai/Voxtral-Mini-4B-Realtime-2602}"
HOST="${VOXTRAL_HOST:-0.0.0.0}"
PORT="${VOXTRAL_PORT:-8000}"
MAX_MODEL_LEN="${VOXTRAL_MAX_MODEL_LEN:-32768}"
GPU_UTIL="${VOXTRAL_GPU_MEMORY_UTILIZATION:-0.5}"

exec vllm serve "${MODEL}" \
  --compilation_config '{"cudagraph_mode":"PIECEWISE"}' \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}"
