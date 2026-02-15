"""Tests for low-latency AdaptiveTextAggregator."""

import asyncio

from voice_agent.pipeline.tts_chunker import AdaptiveTextAggregator, ChunkerConfig


async def _collect(aggregator: AdaptiveTextAggregator, text: str) -> list[str]:
    chunks: list[str] = []
    async for aggregation in aggregator.aggregate(text):
        chunks.append(aggregation.text)
    return chunks


class TestAdaptiveTextAggregator:
    async def test_punctuation_flush(self):
        agg = AdaptiveTextAggregator(ChunkerConfig(min_chars=5, max_chars=120, max_wait_ms=350))
        chunks = []
        chunks.extend(await _collect(agg, "Hello there. "))
        chunks.extend(await _collect(agg, "How are you?"))
        assert chunks[0] == "Hello there."

    async def test_no_punctuation_below_min_chars(self):
        agg = AdaptiveTextAggregator(ChunkerConfig(min_chars=8, max_chars=120, max_wait_ms=350))
        chunks = await _collect(agg, "Hi.")
        assert chunks == []

    async def test_max_chars_flush(self):
        agg = AdaptiveTextAggregator(ChunkerConfig(min_chars=100, max_chars=10, max_wait_ms=350))
        chunks = await _collect(agg, "abcdefghijklmnop")
        assert chunks[0] == "abcdefghij"

    async def test_max_wait_flush(self):
        agg = AdaptiveTextAggregator(ChunkerConfig(min_chars=100, max_chars=120, max_wait_ms=50))
        chunks = await _collect(agg, "hello")
        assert chunks == []
        await asyncio.sleep(0.06)
        chunks = await _collect(agg, " ")
        assert chunks == ["hello"]

    async def test_flush_returns_remaining(self):
        agg = AdaptiveTextAggregator(ChunkerConfig(min_chars=100, max_chars=120, max_wait_ms=350))
        await _collect(agg, "partial text")
        remaining = await agg.flush()
        assert remaining is not None
        assert remaining.text == "partial text"
