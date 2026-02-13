"""Structured logging with session/turn context fields."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Emit JSON-structured log lines with voice-agent context fields."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach voice-agent context if present
        for field in (
            "session_id",
            "room",
            "participant",
            "turn_id",
            "generation_id",
            "event",
            "error_code",
        ):
            val = getattr(record, field, None)
            if val is not None:
                entry[field] = val

        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with structured JSON output."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(StructuredFormatter())
    root.addHandler(handler)

    # Quieten noisy libraries
    for lib in ("urllib3", "httpcore", "httpx", "aiohttp", "websockets"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for the voice agent."""
    return logging.getLogger(f"voice_agent.{name}")
