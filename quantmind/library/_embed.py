"""Private OpenAI embedding implementation."""

from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI


class _OpenAIEmbeddingProvider:
    """Generate embeddings without exposing provider response types."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        client = self._client
        if client is None:
            client = AsyncOpenAI()
            self._client = client
        kwargs: dict[str, Any] = {
            "input": list(texts),
            "model": model,
            "encoding_format": "float",
        }
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        response = await client.embeddings.create(**kwargs)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
