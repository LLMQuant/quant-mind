"""Opinionated SQLite and NumPy semantic knowledge library."""

import asyncio
from pathlib import Path
from uuid import UUID

from typing_extensions import Self

from quantmind.knowledge import BaseKnowledge, TreeKnowledge
from quantmind.library._internal.exact_cosine import (
    _coerce_provider_vectors,
    _decode_stored_vector,
    _encode_vector,
    _ExactCosineIndex,
)
from quantmind.library._internal.index_embeddings import (
    _EmbeddingProvider,
    _OpenAIEmbeddingProvider,
)
from quantmind.library._internal.retrieval_targets import (
    _PROJECTION_SCHEMA_VERSION,
    _project_knowledge,
    _RetrievalTarget,
)
from quantmind.library._internal.sqlite_store import _SQLiteStore
from quantmind.library._types import SemanticHit, SemanticQuery


class LocalKnowledgeLibrary:
    """Persist and semantically search canonical QuantMind knowledge locally."""

    def __init__(
        self,
        store: _SQLiteStore,
        *,
        embedding_model: str,
        embedding_dimensions: int | None,
        embedding_provider: _EmbeddingProvider,
    ) -> None:
        self._store: _SQLiteStore | None = store
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._embedding_provider = embedding_provider
        self._lock = asyncio.Lock()
        self._index: _ExactCosineIndex | None = None

    @classmethod
    async def open(
        cls,
        path: str | Path,
        *,
        embedding_model: str,
        embedding_dimensions: int | None = None,
        _embedding_provider: _EmbeddingProvider | None = None,
    ) -> Self:
        """Open a local library without performing network I/O.

        Args:
            path: SQLite database path or ``":memory:"``.
            embedding_model: Provider model identity recorded with every vector.
            embedding_dimensions: Optional requested vector dimension.

        Returns:
            An open local library.
        """
        if not embedding_model.strip():
            raise ValueError("embedding_model must not be blank")
        if embedding_dimensions is not None and embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be positive")
        return cls(
            _SQLiteStore.open(path),
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            embedding_provider=(
                _embedding_provider or _OpenAIEmbeddingProvider()
            ),
        )

    async def put(self, item: BaseKnowledge) -> None:
        """Atomically persist canonical knowledge and its affected targets."""
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            prepared = store.prepare_put(item)
            targets = _project_knowledge(item)
            existing = prepared.existing_embeddings
            affected: list[_RetrievalTarget] = []
            vectors: dict[str, tuple[bytes, int]] = {}
            for target in targets:
                stored = existing.get(target.target_id)
                needs_embedding = stored is None
                if stored is not None:
                    needs_embedding = any(
                        (
                            stored.embedding_model != self._embedding_model,
                            stored.projection_hash != target.projection_hash,
                            stored.source_content_hash
                            != item.source.content_hash,
                            stored.knowledge_schema_version
                            != item.schema_version,
                            stored.projection_schema_version
                            != _PROJECTION_SCHEMA_VERSION,
                            self._embedding_dimensions is not None
                            and stored.dimension != self._embedding_dimensions,
                        )
                    )
                if needs_embedding:
                    affected.append(target)
                    continue
                assert stored is not None
                _decode_stored_vector(
                    stored.embedding,
                    stored.dimension,
                    target.target_id,
                )
                vectors[target.target_id] = (
                    stored.embedding,
                    stored.dimension,
                )

            if affected:
                provider_values = await self._embedding_provider.embed(
                    [target.text for target in affected],
                    model=self._embedding_model,
                    dimensions=self._embedding_dimensions,
                )
                generated = _coerce_provider_vectors(
                    provider_values,
                    expected_count=len(affected),
                    expected_dimensions=self._embedding_dimensions,
                )
                for target, vector in zip(affected, generated, strict=True):
                    vectors[target.target_id] = _encode_vector(vector)

            store.put(
                prepared,
                targets,
                vectors,
                embedding_model=self._embedding_model,
            )
            self._index = None

    async def get(self, item_id: UUID) -> BaseKnowledge:
        """Return validated canonical knowledge or report not-found/stale data."""
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            return store.get(item_id)

    async def search(self, query: SemanticQuery) -> list[SemanticHit]:
        """Rank filtered targets with deterministic exact cosine similarity."""
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            if self._index is None:
                self._index = _ExactCosineIndex.build(
                    store.load_index_records(
                        embedding_model=self._embedding_model,
                        embedding_dimensions=self._embedding_dimensions,
                    )
                )
            candidates = self._index.filter(query)
            if candidates is None:
                return []

            provider_values = await self._embedding_provider.embed(
                [query.text],
                model=self._embedding_model,
                dimensions=self._embedding_dimensions,
            )
            query_vectors = _coerce_provider_vectors(
                provider_values,
                expected_count=1,
                expected_dimensions=self._embedding_dimensions,
            )
            ranked = candidates.rank(query_vectors[0], top_k=query.top_k)

            canonical: dict[UUID, BaseKnowledge] = {}
            hits: list[SemanticHit] = []
            for result in ranked:
                record = result.record
                item = canonical.get(record.item_id)
                if item is None:
                    try:
                        item = store.get(record.item_id)
                    except KeyError as exc:
                        raise RuntimeError(
                            f"Stale index data for item '{record.item_id}': "
                            "canonical knowledge is missing"
                        ) from exc
                    canonical[record.item_id] = item
                if isinstance(item, TreeKnowledge):
                    if record.node_id is None:
                        citations = item.root().citations
                    else:
                        node = item.nodes.get(record.node_id)
                        if node is None:
                            raise RuntimeError(
                                f"Stale index data for target "
                                f"'{record.target_id}': canonical node is missing"
                            )
                        citations = node.citations
                else:
                    if record.node_id is not None:
                        raise RuntimeError(
                            f"Stale index data for target '{record.target_id}': "
                            "node target belongs to non-tree knowledge"
                        )
                    citations = item.citations
                hits.append(
                    SemanticHit(
                        item_id=record.item_id,
                        node_id=record.node_id,
                        item_type=item.item_type,
                        score=result.score,
                        matched_text=record.matched_text,
                        as_of=item.as_of,
                        available_at=item.available_at,
                        source=item.source,
                        citations=citations,
                    )
                )
            return hits

    async def delete(self, item_id: UUID) -> None:
        """Transactionally remove canonical knowledge and every child record."""
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            store.delete(item_id)
            self._index = None

    async def close(self) -> None:
        """Close SQLite and provider-owned resources; repeated calls are safe."""
        async with self._lock:
            store = self._store
            if store is None:
                return
            self._store = None
            self._index = None
            try:
                await self._embedding_provider.close()
            finally:
                store.close()
