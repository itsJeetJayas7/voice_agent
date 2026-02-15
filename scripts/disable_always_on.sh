#!/usr/bin/env bash
set -euo pipefail

CONF_PATH="/etc/supervisor/conf.d/voice_stack.conf"

if ! command -v supervisorctl >/dev/null 2>&1; then
  echo "supervisorctl is not installed." >&2
  exit 1
fi

if [ -f "${CONF_PATH}" ]; then
  supervisorctl stop voice_stack:* >/dev/null 2>&1 || true
  rm -f "${CONF_PATH}"
  supervisorctl reread >/dev/null
  supervisorctl update >/dev/null
  echo "Always-on mode disabled."
else
  echo "No always-on supervisor config found at ${CONF_PATH}."
fi
