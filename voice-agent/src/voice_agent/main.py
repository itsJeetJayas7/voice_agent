"""Voice Agent main entrypoint.

Bootstraps the agent: loads config, creates session, builds pipeline,
and runs the Pipecat pipeline connected to LiveKit.
"""

from __future__ import annotations

import asyncio

from voice_agent.config import get_settings
from voice_agent.lifecycle import on_first_participant_joined
from voice_agent.logging import get_logger, setup_logging
from voice_agent.pipeline.factory import build_pipeline
from voice_agent.session_manager import SessionManager


async def main() -> None:
    """Main agent entrypoint."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("main")

    logger.info("Voice Agent starting up")
    logger.info("LiveKit URL: %s", settings.livekit_url)
    logger.info("Cerebras model: %s", settings.cerebras_model)
    logger.info("Max concurrent sessions: %d", settings.max_concurrent_sessions)

    # ── Session management ───────────────────────────────
    session_manager = SessionManager(max_concurrent=settings.max_concurrent_sessions)

    try:
        # Create a session for this agent instance
        session = await session_manager.create(
            room=settings.agent_room,
            identity=settings.agent_identity,
        )

        # ── Generate agent token ─────────────────────────
        from livekit.api import AccessToken, VideoGrants
        from datetime import timedelta

        grants = VideoGrants(
            room_join=True,
            room=settings.agent_room,
            can_publish=True,
            can_subscribe=True,
        )

        token = (
            AccessToken(
                api_key=settings.livekit_api_key,
                api_secret=settings.livekit_api_secret,
            )
            .with_identity(settings.agent_identity)
            .with_grants(grants)
            .with_ttl(timedelta(hours=24))
            .to_jwt()
        )

        # ── Build and run pipeline ───────────────────────
        pipeline, task, runner, interruption = build_pipeline(
            settings=settings,
            session=session,
            livekit_url=settings.livekit_url,
            livekit_token=token,
            room_name=settings.agent_room,
        )

        # Register lifecycle events

        # The transport is the first processor in the pipeline.
        # We access it to register event handlers.
        transport_input = pipeline.processors[0]
        if hasattr(transport_input, '_transport'):
            transport = transport_input._transport

            @transport.event_handler("on_first_participant_joined")  # type: ignore[untyped-decorator]
            async def _on_first(transport_obj: object, participant_id: str) -> None:
                await on_first_participant_joined(transport_obj, participant_id, task)

        logger.info(
            "Pipeline ready — waiting for participants in room '%s'",
            settings.agent_room,
            extra={
                "session_id": session.session_id,
                "room": settings.agent_room,
                "event": "agent_ready",
            },
        )

        await runner.run(task)

    except KeyboardInterrupt:
        logger.info("Shutting down (keyboard interrupt)")
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
    finally:
        await session_manager.close_all()
        logger.info("Voice Agent shut down")


if __name__ == "__main__":
    asyncio.run(main())
