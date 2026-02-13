"""Tests for the TTS chunker — adaptive speakable chunk splitting."""

import asyncio

import pytest

from voice_agent.pipeline.tts_chunker import TTSChunker, ChunkerConfig


@pytest.fixture
def chunker():
    """Create a chunker with default config."""
    return TTSChunker(ChunkerConfig(min_chars=24, max_chars=120, max_wait_ms=350))


@pytest.fixture
def small_chunker():
    """Create a chunker with small thresholds for testing."""
    return TTSChunker(ChunkerConfig(min_chars=5, max_chars=20, max_wait_ms=100))


class TestPunctuationFlush:
    """Test that chunks flush at punctuation boundaries."""

    async def test_period_flush(self, chunker: TTSChunker):
        """Flush at period when min_chars is met."""
        text = "Hello, this is a test sentence."
        for char in text:
            await chunker.add_token(char)

        queue = await chunker.chunks()
        chunk = queue.get_nowait()
        assert chunk.endswith(".")
        assert len(chunk) >= chunker.config.min_chars

    async def test_question_mark_flush(self, chunker: TTSChunker):
        """Flush at question mark when min_chars is met."""
        text = "How are you doing today? I hope well."
        for char in text:
            await chunker.add_token(char)

        queue = await chunker.chunks()
        chunk = queue.get_nowait()
        assert "?" in chunk

    async def test_comma_flush(self, chunker: TTSChunker):
        """Flush at comma when min_chars is met."""
        text = "In the beginning of time, there was nothing but silence."
        for char in text:
            await chunker.add_token(char)

        queue = await chunker.chunks()
        chunk = queue.get_nowait()
        assert "," in chunk

    async def test_no_flush_below_min_chars(self, small_chunker: TTSChunker):
        """Don't flush at punctuation if below min_chars."""
        await small_chunker.add_token("Hi.")
        queue = await small_chunker.chunks()
        assert queue.empty()


class TestMaxCharFlush:
    """Test that chunks flush at max_chars boundary."""

    async def test_max_chars_flush(self, small_chunker: TTSChunker):
        """Force flush when max_chars is reached."""
        # Add 25 chars to a chunker with max_chars=20
        text = "a" * 25
        for char in text:
            await small_chunker.add_token(char)

        queue = await small_chunker.chunks()
        chunk = queue.get_nowait()
        assert len(chunk) <= small_chunker.config.max_chars + 1  # +1 for the triggering char


class TestMaxWaitFlush:
    """Test that chunks flush after max_wait_ms."""

    async def test_max_wait_flush(self):
        """Flush after max_wait_ms even without punctuation."""
        chunker = TTSChunker(ChunkerConfig(min_chars=5, max_chars=120, max_wait_ms=50))

        await chunker.add_token("Hello world")
        # Wait for max_wait_ms to elapse
        await asyncio.sleep(0.06)
        # Trigger check by adding another token
        await chunker.add_token(" ")

        queue = await chunker.chunks()
        assert not queue.empty()


class TestFlushRemaining:
    """Test residual text flushing on stream end."""

    async def test_flush_remaining(self, chunker: TTSChunker):
        """Flush remaining text when LLM stream ends."""
        await chunker.add_token("Short text")
        await chunker.flush_remaining()

        queue = await chunker.chunks()
        chunk = queue.get_nowait()
        assert chunk == "Short text"
        # End-of-stream marker
        end = queue.get_nowait()
        assert end == ""

    async def test_flush_remaining_empty(self, chunker: TTSChunker):
        """Flush remaining does nothing on empty buffer."""
        await chunker.flush_remaining()
        queue = await chunker.chunks()
        # Only the end-of-stream marker
        end = queue.get_nowait()
        assert end == ""


class TestCancellation:
    """Test cancellation behaviour."""

    async def test_cancel_clears_buffer(self, chunker: TTSChunker):
        """Cancellation clears accumulated text."""
        await chunker.add_token("Some text before cancel")
        chunker.cancel()
        assert chunker._buffer == ""

    async def test_cancel_drains_queue(self, chunker: TTSChunker):
        """Cancellation drains the output queue."""
        await chunker.add_token("a" * 200)  # Force flush
        chunker.cancel()
        queue = await chunker.chunks()
        assert queue.empty()

    async def test_add_after_cancel_is_noop(self, chunker: TTSChunker):
        """Adding tokens after cancel is a no-op."""
        chunker.cancel()
        await chunker.add_token("ignored")
        assert chunker._buffer == ""


class TestReset:
    """Test reset behaviour."""

    async def test_reset_clears_state(self, chunker: TTSChunker):
        """Reset clears buffer and queue."""
        await chunker.add_token("text before reset")
        chunker.reset()
        assert chunker._buffer == ""
        assert chunker._first_token_time == 0.0
        queue = await chunker.chunks()
        assert queue.empty()
