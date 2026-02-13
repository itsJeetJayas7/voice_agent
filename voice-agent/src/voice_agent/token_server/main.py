"""FastAPI token server for LiveKit join tokens."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants

from voice_agent.config import get_settings
from voice_agent.logging import get_logger, setup_logging
from voice_agent.token_server.schemas import TokenRequest, TokenResponse

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger("token_server")

app = FastAPI(
    title="Voice Agent Token Server",
    version="0.1.0",
    docs_url="/docs",
)

# CORS — restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/token", response_model=TokenResponse)
async def create_token(req: TokenRequest) -> TokenResponse:
    """Issue a LiveKit join token for a user or agent participant."""
    try:
        ttl = timedelta(seconds=settings.token_ttl_seconds)
        expires_at = datetime.now(timezone.utc) + ttl

        grants = VideoGrants(
            room_join=True,
            room=req.room,
            can_publish=True,
            can_subscribe=True,
        )

        token = (
            AccessToken(
                api_key=settings.livekit_api_key,
                api_secret=settings.livekit_api_secret,
            )
            .with_identity(req.identity)
            .with_grants(grants)
            .with_ttl(ttl)
            .to_jwt()
        )

        logger.info(
            "Token issued for %s/%s (%s)",
            req.room,
            req.identity,
            req.participant_type,
            extra={
                "room": req.room,
                "participant": req.identity,
                "event": "token_issued",
            },
        )

        return TokenResponse(
            token=token,
            ws_url=settings.livekit_url,
            room=req.room,
            identity=req.identity,
            expires_at=expires_at,
        )

    except Exception as exc:
        logger.error("Token creation failed: %s", exc, extra={"error_code": "token_error"})
        raise HTTPException(status_code=500, detail="Token creation failed") from exc
