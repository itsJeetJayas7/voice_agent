"""Pipeline factory — composes the full Pipecat voice-agent pipeline.

Pipeline order:
1. transport.input()       — Receive user audio via LiveKit
2. STT (Voxtral)           — Speech → TranscriptionFrame
3. User context aggregator — Collect user turns (with VAD)
4. LLM (Cerebras)          — Generate streaming text response
5. TTS (Chatterbox)        — Text → audio frames
6. transport.output()      — Publish agent audio via LiveKit
7. Assistant aggregator    — Collect assistant turns for context
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from voice_agent.config import Settings
from voice_agent.logging import get_logger
from voice_agent.pipeline.interruption_controller import InterruptionController
from voice_agent.pipeline.transcript_broadcaster import TranscriptBroadcaster
from voice_agent.pipeline.tts_chunker import AdaptiveTextAggregator, ChunkerConfig
from voice_agent.pipeline.voxtral_realtime_stt_service import VoxtralRealtimeSTTService

if TYPE_CHECKING:
    from voice_agent.session import Session

logger = get_logger("pipeline.factory")

# System prompt for the voice agent
SYSTEM_PROMPT = (
    "You are a helpful voice assistant in a real-time conversation. "
    "Your responses will be spoken aloud, so keep them natural, concise, and conversational. "
    "Avoid emojis, markdown, bullet points, or formatting that cannot be spoken. "
    "You may use Chatterbox expression tags sparingly when they improve delivery: "
    "[laugh], [chuckle], [sigh], [cough], [gasp], [groan], [sniff]. "
    "Keep tags inline and avoid overusing them. "
    "Respond clearly and helpfully to whatever the user says."
)


def build_pipeline(
    settings: Settings,
    session: Session,
    livekit_url: str,
    livekit_token: str,
    room_name: str,
) -> tuple[Pipeline, PipelineTask, PipelineRunner, InterruptionController]:
    """Compose the full Pipecat voice-agent pipeline.

    Returns:
        Tuple of (pipeline, task, runner, interruption_controller).
    """
    # ── Transport ────────────────────────────────────────
    transport = LiveKitTransport(
        url=livekit_url,
        token=livekit_token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    # ── VAD + STT (Voxtral Realtime WS API) ───────────────
    # Emit VAD start/stop frames before STT so we can commit utterances cleanly.
    vad_params = VADParams(
        confidence=settings.vad_confidence,
        start_secs=settings.vad_start_secs,
        stop_secs=settings.vad_stop_secs,
        min_volume=settings.vad_min_volume,
    )
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=vad_params))

    stt = VoxtralRealtimeSTTService(
        host=settings.voxtral_host,
        port=settings.voxtral_port,
        model="mistralai/Voxtral-Mini-4B-Realtime-2602",
        fallback_commit_enabled=settings.stt_fallback_commit_enabled,
        fallback_commit_interval_s=settings.stt_fallback_commit_interval_s,
        fallback_min_voiced_appends=settings.stt_fallback_min_voiced_appends,
        voiced_peak_threshold=settings.stt_voiced_peak_threshold,
        vad_stop_debounce_ms=settings.stt_vad_stop_debounce_ms,
    )

    # ── LLM (Cerebras via OpenAI-compatible API) ─────────
    llm = OpenAILLMService(
        api_key=settings.cerebras_api_key,
        base_url=settings.cerebras_base_url,
        model=settings.cerebras_model,
    )

    # ── TTS (Chatterbox via OpenAI-compatible API) ───────
    tts = OpenAITTSService(
        api_key="not-needed",
        base_url=settings.chatterbox_base_url + "/v1",
        model="chatterbox-turbo",
        voice="alloy",
        sample_rate=settings.chatterbox_sample_rate,
    )
    tts_text_processor = LLMTextProcessor(
        text_aggregator=AdaptiveTextAggregator(ChunkerConfig.from_settings(settings))
    )

    # ── Context management ───────────────────────────────
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages)  # type: ignore[arg-type]

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # ── Transcript broadcasters ────────────────────────────
    # Two instances: one to capture user transcripts (after STT),
    # one to capture agent text (after LLM).
    user_broadcaster = TranscriptBroadcaster(
        transport,
        capture_user=True,
        capture_agent=False,
    )
    agent_broadcaster = TranscriptBroadcaster(
        transport,
        capture_user=False,
        capture_agent=True,
    )

    # ── Pipeline composition ─────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),           # Receive user audio
            vad,                         # Local VAD start/stop frames
            stt,                         # Voxtral: audio → text
            user_broadcaster,            # Broadcast user transcripts
            user_aggregator,             # Collect user context (with VAD)
            llm,                         # Cerebras LLM (streaming)
            agent_broadcaster,           # Broadcast agent text
            tts_text_processor,          # Low-latency LLM text chunking
            tts,                         # Chatterbox: text → audio
            transport.output(),          # Send agent audio
            assistant_aggregator,        # Collect assistant context
        ]
    )

    task = PipelineTask(
        pipeline,
        cancel_on_idle_timeout=settings.agent_cancel_on_idle_timeout,
        idle_timeout_secs=settings.agent_idle_timeout_secs,
        params=PipelineParams(
            enable_metrics=settings.metrics_enabled,
            enable_usage_metrics=settings.metrics_enabled,
        ),
    )

    runner = PipelineRunner()

    # ── Interruption controller ──────────────────────────
    interruption = InterruptionController(session)

    logger.info(
        "Pipeline built for room %s",
        room_name,
        extra={"session_id": session.session_id, "room": room_name},
    )

    return pipeline, task, runner, interruption
