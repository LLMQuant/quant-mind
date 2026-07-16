"""Private side-effect seams for the local knowledge library."""

from collections.abc import Sequence
from typing import Protocol


class _EmbeddingProvider(Protocol):
    """Private embedding seam used by production code and deterministic tests."""

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> Sequence[Sequence[float]]:
        """Embed texts in input order."""
        ...

    async def close(self) -> None:
        """Release provider-owned resources."""
        ...
