"""Centralised configuration via pydantic-settings + .env."""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All knobs live here.  Loaded from environment / .env in project root."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LiveKit ──────────────────────────────────────────
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "devsecret"

    # ── Cerebras LLM ────────────────────────────────────
    cerebras_api_key: str = ""
    cerebras_model: str = "gpt-oss-120b"
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_max_tokens: int = 1024
    cerebras_timeout_s: float = 30.0
    cerebras_max_retries: int = 3

    # ── Voxtral STT ─────────────────────────────────────
    voxtral_protocol: Literal["http", "ws", "grpc"] = "http"
    voxtral_host: str = "localhost"
    voxtral_port: int = 8000
    voxtral_path: str = "/v1/audio/transcriptions"
    voxtral_timeout_ms: int = 5000

    # ── Chatterbox TTS ──────────────────────────────────
    chatterbox_protocol: Literal["http", "ws", "grpc"] = "http"
    chatterbox_host: str = "localhost"
    chatterbox_port: int = 8001
    chatterbox_path: str = "/v1/audio/speech"
    chatterbox_timeout_ms: int = 5000
    chatterbox_voice: str = "default"
    chatterbox_sample_rate: int = 24000

    # ── Agent ────────────────────────────────────────────
    agent_room: str = "voice-room"
    agent_identity: str = "voice-agent"
    max_concurrent_sessions: int = 10

    # ── VAD / Endpointing ───────────────────────────────
    vad_confidence: float = 0.6
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.8
    vad_min_volume: float = 0.001
    endpoint_min_silence_ms: int = 300
    endpoint_max_silence_ms: int = 1500
    max_utterance_secs: int = 30

    # ── TTS Chunker ─────────────────────────────────────
    chunk_min_chars: int = 24
    chunk_max_chars: int = 120
    chunk_max_wait_ms: int = 350

    # ── Audio ────────────────────────────────────────────
    audio_frame_ms: int = 20
    stt_send_chunk_ms: int = 100
    tts_output_frame_ms: int = 20

    # ── Logging / Metrics ───────────────────────────────
    log_level: str = "INFO"
    log_transcripts: bool = False
    metrics_enabled: bool = True

    # ── Token Server ─────────────────────────────────────
    token_server_host: str = "0.0.0.0"
    token_server_port: int = 8081
    token_ttl_seconds: int = 3600

    # ── Derived helpers ──────────────────────────────────
    @property
    def voxtral_base_url(self) -> str:
        scheme = "http" if self.voxtral_protocol == "http" else "http"
        return f"{scheme}://{self.voxtral_host}:{self.voxtral_port}"

    @property
    def voxtral_ws_url(self) -> str:
        return f"ws://{self.voxtral_host}:{self.voxtral_port}{self.voxtral_path}"

    @property
    def chatterbox_base_url(self) -> str:
        return f"http://{self.chatterbox_host}:{self.chatterbox_port}"

    @property
    def chatterbox_ws_url(self) -> str:
        return f"ws://{self.chatterbox_host}:{self.chatterbox_port}{self.chatterbox_path}"

    @field_validator("cerebras_api_key")
    @classmethod
    def _cerebras_key_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "CEREBRAS_API_KEY is required. "
                "Set it in .env or as an environment variable."
            )
        return v


def get_settings() -> Settings:
    """Singleton-ish factory; import and call where needed."""
    return Settings()
