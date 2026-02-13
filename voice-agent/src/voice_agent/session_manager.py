"""Session manager: concurrent session tracking keyed by room + identity."""

from __future__ import annotations

import asyncio
from typing import Optional

from voice_agent.logging import get_logger
from voice_agent.session import Session

logger = get_logger("session_manager")


class SessionManager:
    """Manages concurrent voice-agent sessions with a hard cap.

    Keyed by ``(room, identity)`` pairs.  No global mutable state
    leaks between sessions.
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        self._max = max_concurrent
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(room: str, identity: str) -> str:
        return f"{room}::{identity}"

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def create(self, room: str, identity: str) -> Session:
        """Create and register a new session.  Raises if at capacity."""
        async with self._lock:
            key = self._key(room, identity)

            # Close existing session for same room+identity (reconnect case)
            if key in self._sessions:
                old = self._sessions.pop(key)
                logger.warning(
                    "Replacing existing session %s for %s",
                    old.session_id,
                    key,
                    extra={"session_id": old.session_id, "room": room},
                )
                await old.close()

            if len(self._sessions) >= self._max:
                raise RuntimeError(
                    f"Max concurrent sessions ({self._max}) reached. "
                    "Reject or queue the connection."
                )

            session = Session(room=room, identity=identity)
            self._sessions[key] = session
            logger.info(
                "Session created: %s (%s)",
                session.session_id,
                key,
                extra={
                    "session_id": session.session_id,
                    "room": room,
                    "event": "session_created",
                },
            )
            return session

    async def get(self, room: str, identity: str) -> Optional[Session]:
        """Return the session for the given room+identity, or None."""
        return self._sessions.get(self._key(room, identity))

    async def remove(self, room: str, identity: str) -> None:
        """Close and deregister the session."""
        async with self._lock:
            key = self._key(room, identity)
            session = self._sessions.pop(key, None)
        if session:
            await session.close()
            logger.info(
                "Session removed: %s (%s)",
                session.session_id,
                key,
                extra={
                    "session_id": session.session_id,
                    "room": room,
                    "event": "session_removed",
                },
            )

    async def close_all(self) -> None:
        """Shut down all sessions (for graceful process exit)."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            await s.close()
        logger.info("All sessions closed", extra={"event": "all_sessions_closed"})
