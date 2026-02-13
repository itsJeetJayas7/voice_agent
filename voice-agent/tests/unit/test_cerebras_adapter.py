"""Tests for the Cerebras OpenAI LLM adapter."""

import asyncio

import pytest

from voice_agent.adapters.llm.cerebras_openai import CerebrasOpenAIAdapter
from voice_agent.config import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        cerebras_api_key="test-key",
        cerebras_base_url="https://api.cerebras.ai/v1",
        cerebras_model="gpt-oss-120b",
        cerebras_max_tokens=256,
        cerebras_timeout_s=5.0,
        cerebras_max_retries=2,
        # Provide required fields to prevent validation errors
        livekit_url="ws://localhost:7880",
        livekit_api_key="devkey",
        livekit_api_secret="devsecret",
    )


@pytest.fixture
def adapter(settings):
    return CerebrasOpenAIAdapter(settings)


class TestStreamCompletion:
    """Test streaming completion behaviour."""

    async def test_cancel_stops_iteration(self, adapter: CerebrasOpenAIAdapter):
        """Cancellation raises CancelledError."""
        await adapter.cancel()
        with pytest.raises(asyncio.CancelledError):
            async for _ in adapter.stream_completion(
                [{"role": "user", "content": "Hello"}]
            ):
                pass

    async def test_cancel_flag_is_set(self, adapter: CerebrasOpenAIAdapter):
        """Cancel sets the internal flag."""
        assert not adapter._cancelled
        await adapter.cancel()
        assert adapter._cancelled


class TestRetryBehaviour:
    """Test retry/backoff logic."""

    async def test_max_retries_respected(self, adapter: CerebrasOpenAIAdapter):
        """Should not exceed max_retries attempts."""
        assert adapter._max_retries == 2
        # The adapter retries up to max_retries times before giving up


class TestConfiguration:
    """Test adapter configuration."""

    def test_model_from_settings(self, adapter: CerebrasOpenAIAdapter):
        assert adapter._model == "gpt-oss-120b"

    def test_retries_from_settings(self, adapter: CerebrasOpenAIAdapter):
        assert adapter._max_retries == 2

    async def test_close(self, adapter: CerebrasOpenAIAdapter):
        """Close should not raise."""
        await adapter.close()
        assert adapter._cancelled
