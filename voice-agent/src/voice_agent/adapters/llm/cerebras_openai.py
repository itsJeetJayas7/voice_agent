"""Cerebras LLM adapter — OpenAI-compatible streaming via AsyncOpenAI.

Uses the Cerebras inference API which is OpenAI-compatible but with
specific differences (see Cerebras docs).
"""

from __future__ import annotations

import asyncio
import random
from typing import AsyncIterator

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, APIStatusError

from voice_agent.adapters.llm.base import LLMToken
from voice_agent.config import Settings
from voice_agent.logging import get_logger

logger = get_logger("llm.cerebras")


class CerebrasOpenAIAdapter:
    """Cerebras LLM adapter using AsyncOpenAI with streaming.

    Features:
    - Exponential backoff with jitter on retryable errors (429, 5xx, connection)
    - Cancellation preempts retry attempts
    - Avoids unsupported OpenAI fields for Cerebras compatibility
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.cerebras_api_key,
            base_url=settings.cerebras_base_url,
            timeout=settings.cerebras_timeout_s,
            max_retries=0,  # We handle retries ourselves for cancellation support
        )
        self._model = settings.cerebras_model
        self._max_retries = settings.cerebras_max_retries
        self._cancelled = False

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[LLMToken]:
        """Stream tokens from Cerebras with retry/backoff.

        Yields LLMToken objects.  Cancellation-aware: checks self._cancelled
        and raises CancelledError if set.
        """
        max_tokens = max_tokens or self._settings.cerebras_max_tokens

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if self._cancelled:
                raise asyncio.CancelledError("LLM request cancelled")

            try:
                # Cerebras does not support: frequency_penalty, presence_penalty,
                # logit_bias, logprobs, top_logprobs, n > 1, stop sequences
                stream = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )

                async for chunk in stream:  # type: ignore[union-attr]
                    if self._cancelled:
                        # Close the stream and bail
                        await stream.close()
                        raise asyncio.CancelledError("LLM stream cancelled")

                    for choice in chunk.choices:
                        delta = choice.delta
                        if delta and delta.content:
                            yield LLMToken(
                                text=delta.content,
                                finish_reason=choice.finish_reason,
                            )
                        elif choice.finish_reason:
                            yield LLMToken(
                                text="",
                                finish_reason=choice.finish_reason,
                            )
                return  # Success — exit retry loop

            except asyncio.CancelledError:
                raise
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                logger.warning(
                    "Cerebras connection/timeout error (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
            except APIStatusError as exc:
                last_exc = exc
                status = exc.status_code
                if status in (408, 429) or status >= 500:
                    logger.warning(
                        "Cerebras retryable error %d (attempt %d/%d): %s",
                        status,
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                    )
                else:
                    # Non-retryable error (4xx other than 408/429)
                    logger.error("Cerebras non-retryable error %d: %s", status, exc)
                    raise

            # Exponential backoff with jitter
            if attempt < self._max_retries:
                backoff = min(2**attempt + random.uniform(0, 1), 10.0)
                logger.info("Retrying in %.1fs...", backoff)

                # Check cancellation during backoff
                try:
                    await asyncio.wait_for(
                        self._wait_cancel(), timeout=backoff
                    )
                    # If wait_cancel returned, we were cancelled
                    raise asyncio.CancelledError("LLM cancelled during backoff")
                except asyncio.TimeoutError:
                    pass  # Backoff elapsed, proceed to retry

        # Exhausted retries
        if last_exc:
            raise last_exc

    async def _wait_cancel(self) -> None:
        """Wait until cancellation is signalled."""
        while not self._cancelled:
            await asyncio.sleep(0.05)

    async def cancel(self) -> None:
        """Cancel the current streaming request."""
        self._cancelled = True
        logger.debug("Cerebras LLM request cancelled")

    async def close(self) -> None:
        """Close the AsyncOpenAI client."""
        self._cancelled = True
        await self._client.close()
        logger.info("Cerebras LLM adapter closed")
