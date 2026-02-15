#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_DIR="${ROOT_DIR}/voice-agent"
LOG_DIR="${ROOT_DIR}/logs"
CONF_PATH="/etc/supervisor/conf.d/voice_stack.conf"

if ! command -v supervisorctl >/dev/null 2>&1; then
  echo "supervisorctl is not installed. Install supervisor first." >&2
  exit 1
fi

if [ ! -f "${VOICE_DIR}/.venv/bin/activate" ]; then
  echo "Missing ${VOICE_DIR}/.venv. Create it before enabling always-on mode." >&2
  exit 1
fi

if [ ! -f "${ROOT_DIR}/start_vllm.sh" ]; then
  echo "Missing ${ROOT_DIR}/start_vllm.sh" >&2
  exit 1
fi

if [ ! -f "${ROOT_DIR}/scripts/run_voice_agent.sh" ]; then
  echo "Missing ${ROOT_DIR}/scripts/run_voice_agent.sh" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

cat > "${CONF_PATH}" <<'EOF'
[group:voice_stack]
programs=voice_livekit,voice_vllm,voice_tts,voice_token,voice_agent,voice_web

[program:voice_livekit]
directory=/root
command=/bin/bash -lc "exec livekit-server --config /root/voice-agent/infra/livekit/livekit.yaml --dev"
autostart=true
autorestart=true
startsecs=2
startretries=20
stopsignal=TERM
stopasgroup=true
killasgroup=true
priority=100
stdout_logfile=/root/logs/livekit.log
stderr_logfile=/root/logs/livekit.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=5
environment=HOME="/root",PYTHONUNBUFFERED="1"

[program:voice_vllm]
directory=/root
command=/bin/bash -lc "cd /root && exec /root/start_vllm.sh"
autostart=true
autorestart=true
startsecs=5
startretries=20
stopsignal=TERM
stopasgroup=true
killasgroup=true
priority=110
stdout_logfile=/root/logs/vllm.log
stderr_logfile=/root/logs/vllm.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=5
environment=HOME="/root",PYTHONUNBUFFERED="1"

[program:voice_tts]
directory=/root/voice-agent
command=/bin/bash -lc "cd /root/voice-agent && source .venv/bin/activate && exec python -m voice_agent.tts_server"
autostart=true
autorestart=true
startsecs=2
startretries=20
stopsignal=TERM
stopasgroup=true
killasgroup=true
priority=120
stdout_logfile=/root/logs/tts_server.log
stderr_logfile=/root/logs/tts_server.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=5
environment=HOME="/root",PYTHONUNBUFFERED="1"

[program:voice_token]
directory=/root/voice-agent
command=/bin/bash -lc "cd /root/voice-agent && source .venv/bin/activate && exec uvicorn voice_agent.token_server.main:app --host 0.0.0.0 --port 8081"
autostart=true
autorestart=true
startsecs=2
startretries=20
stopsignal=TERM
stopasgroup=true
killasgroup=true
priority=130
stdout_logfile=/root/logs/token_server.log
stderr_logfile=/root/logs/token_server.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=5
environment=HOME="/root",PYTHONUNBUFFERED="1"

[program:voice_agent]
directory=/root/voice-agent
command=/bin/bash -lc "cd /root && exec /root/scripts/run_voice_agent.sh"
autostart=true
autorestart=true
startsecs=3
startretries=20
stopsignal=TERM
stopasgroup=true
killasgroup=true
priority=140
stdout_logfile=/root/logs/agent.log
stderr_logfile=/root/logs/agent.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=5
environment=HOME="/root",PYTHONUNBUFFERED="1"

[program:voice_web]
directory=/root/voice-agent
command=/bin/bash -lc "exec python3 -m http.server 5173 -d /root/voice-agent/web"
autostart=true
autorestart=true
startsecs=1
startretries=20
stopsignal=TERM
stopasgroup=true
killasgroup=true
priority=150
stdout_logfile=/root/logs/web.log
stderr_logfile=/root/logs/web.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=5
environment=HOME="/root",PYTHONUNBUFFERED="1"
EOF

# Stop manually-launched copies to avoid port conflicts before supervisor starts them.
"${ROOT_DIR}/scripts/stop_stack.sh" >/dev/null 2>&1 || true

supervisorctl reread >/dev/null
supervisorctl update >/dev/null
supervisorctl start voice_stack:* >/dev/null || true

echo "Always-on mode enabled via supervisor."
supervisorctl status voice_stack:* || true
