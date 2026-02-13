"""Per-room Session object with task registry, cancellation, and cleanup."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from voice_agent.logging import get_logger
from voice_agent.metrics import MetricsCollector, TurnMetrics

logger = get_logger("session")


class SessionState(Enum):
    INITIALIZING = auto()
    ACTIVE = auto()
    INTERRUPTING = auto()
    CLOSING = auto()
    CLOSED = auto()


@dataclass
class Session:
    """Represents one voice-agent room interaction lifecycle.

    Owns pipeline tasks, adapter clients, cancellation primitives,
    queues, and metrics context.  Guarantees deterministic cleanup.
    """

    room: str
    identity: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: SessionState = SessionState.INITIALIZING

    # Turn management
    turn_id: str = ""
    generation_id: str = ""

    # Cancellation
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Task registry — all long-running tasks registered here for cleanup
    _tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)

    # Metrics
    metrics: MetricsCollector = field(default=None)  # type: ignore[assignment]

    # Adapter / resource references (set externally)
    resources: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = MetricsCollector(session_id=self.session_id)

    # ── Turn lifecycle ────────────────────────────────────

    def new_turn(self) -> TurnMetrics:
        """Begin a new conversation turn; return fresh metrics object."""
        self.turn_id = uuid.uuid4().hex[:8]
        self.generation_id = uuid.uuid4().hex[:8]
        self._cancel_event.clear()
        self.state = SessionState.ACTIVE
        logger.info(
            "New turn started",
            extra={
                "session_id": self.session_id,
                "room": self.room,
                "turn_id": self.turn_id,
                "generation_id": self.generation_id,
                "event": "turn_start",
            },
        )
        return self.metrics.new_turn(self.turn_id, self.generation_id)

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def cancel_turn(self) -> None:
        """Signal cancellation for the current turn."""
        self._cancel_event.set()
        self.state = SessionState.INTERRUPTING
        logger.info(
            "Turn cancelled",
            extra={
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "event": "turn_cancel",
            },
        )

    async def wait_for_cancel(self) -> None:
        """Await the cancellation event."""
        await self._cancel_event.wait()

    def check_cancelled(self) -> None:
        """Raise CancelledError if turn is cancelled. Call in hot loops."""
        if self._cancel_event.is_set():
            raise asyncio.CancelledError("Turn cancelled by interruption")

    # ── Task registry ─────────────────────────────────────

    def register_task(self, name: str, task: asyncio.Task[Any]) -> None:
        self._tasks[name] = task
        logger.debug(
            "Task registered: %s", name,
            extra={"session_id": self.session_id},
        )

    def cancel_tasks(self, *names: str) -> None:
        """Cancel specific named tasks (or all if none specified)."""
        targets = names or tuple(self._tasks.keys())
        for name in targets:
            task = self._tasks.get(name)
            if task and not task.done():
                task.cancel()
                logger.debug(
                    "Task cancelled: %s", name,
                    extra={"session_id": self.session_id},
                )

    # ── Cleanup ───────────────────────────────────────────

    async def close(self) -> None:
        """Deterministic cleanup: cancel tasks, close clients, clear queues."""
        if self.state == SessionState.CLOSED:
            return
        self.state = SessionState.CLOSING
        logger.info(
            "Session closing",
            extra={"session_id": self.session_id, "room": self.room, "event": "session_close"},
        )

        # 1. Cancel all tasks
        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()

        # 2. Await all tasks (suppress CancelledError)
        if self._tasks:
            results = await asyncio.gather(
                *self._tasks.values(), return_exceptions=True
            )
            for name, result in zip(self._tasks.keys(), results):
                if isinstance(result, Exception) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    logger.warning(
                        "Task %s raised during cleanup: %s", name, result,
                        extra={"session_id": self.session_id},
                    )
        self._tasks.clear()

        # 3. Close adapter clients
        for key, resource in self.resources.items():
            if hasattr(resource, "close"):
                try:
                    await resource.close()
                except Exception as exc:
                    logger.warning(
                        "Resource %s close error: %s", key, exc,
                        extra={"session_id": self.session_id},
                    )
        self.resources.clear()

        self.state = SessionState.CLOSED
        logger.info(
            "Session closed",
            extra={"session_id": self.session_id, "room": self.room, "event": "session_closed"},
        )
