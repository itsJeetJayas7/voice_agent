"""Request / response schemas for the token server."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TokenRequest(BaseModel):
    """POST /token request body."""

    room: str = Field(..., min_length=1, max_length=128, description="LiveKit room name")
    identity: str = Field(..., min_length=1, max_length=128, description="Participant identity")
    participant_type: Literal["user", "agent"] = Field(
        default="user", description="Participant type"
    )

    @field_validator("room", "identity")
    @classmethod
    def _sanitize(cls, v: str) -> str:
        """Reject dangerous characters in room/identity names."""
        import re

        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError(
                f"Invalid characters in '{v}'. "
                "Only alphanumeric, underscore, and hyphen are allowed."
            )
        return v


class TokenResponse(BaseModel):
    """POST /token response body."""

    token: str
    ws_url: str
    room: str
    identity: str
    expires_at: datetime
