"""Opinionated SQLite and LlamaIndex semantic knowledge library."""

import asyncio
from pathlib import Path
from uuid import UUID

from typing_extensions import Self

from quantmind.knowledge import (
    ArtifactLocator,
    BaseKnowledge,
    Citation,
    PaperArtifact,
    PaperChunkSet,
    PaperGlobalSummary,
    PaperSemanticResult,
    PaperSourceRevision,
    PaperStructureTree,
    ResolvedPaperArtifact,
    TreeKnowledge,
    TreeNode,
)
from quantmind.library._internal.index_embeddings import (
    _EmbeddingProvider,
    _OpenAIEmbeddingProvider,
)
from quantmind.library._internal.llamaindex_retriever import (
    _coerce_provider_vectors,
    _decode_stored_vector,
    _encode_vector,
    _LlamaIndexRetriever,
)
from quantmind.library._internal.retrieval_targets import (
    _PROJECTION_SCHEMA_VERSION,
    _project_knowledge,
    _project_paper,
    _RetrievalTarget,
)
from quantmind.library._internal.sqlite_store import _SQLiteStore
from quantmind.library._types import (
    SearchProjection,
    SemanticHit,
    SemanticQuery,
)


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
        self._index: _LlamaIndexRetriever | None = None

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

    async def put(self, item: BaseKnowledge | PaperStructureTree) -> None:
        """Atomically persist canonical knowledge or a self-contained tree.

        A ``PaperStructureTree`` is a derived, self-contained artifact: it is
        stored on its own from its provenance metadata (``as_of`` / source ref /
        ``source_content_hash``), with no source revision or chunk set required.
        Every other supported value is canonical ``BaseKnowledge``.
        """
        if isinstance(item, PaperStructureTree):
            await self.put_structure_tree(item)
            return
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

    async def put_paper(self, result: PaperSemanticResult) -> None:
        """Persist one source-first paper result and all required projections.

        Embeddings are prepared before the SQLite transaction. A provider
        failure therefore leaves no partial source, artifact, lineage, or
        required projection records.
        """
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            prepared = store.prepare_put_paper(result)
            targets = _project_paper(result)
            artifacts = {
                result.chunk_set.id: result.chunk_set,
                result.global_summary.id: result.global_summary,
            }
            affected: list[_RetrievalTarget] = []
            vectors: dict[str, tuple[bytes, int]] = {}
            for target in targets:
                stored = prepared.existing_embeddings.get(target.target_id)
                artifact = artifacts[target.artifact_id]
                needs_embedding = stored is None
                if stored is not None:
                    needs_embedding = any(
                        (
                            stored.embedding_model != self._embedding_model,
                            stored.projection_hash != target.projection_hash,
                            stored.source_content_hash
                            != result.source_revision.source.content_hash,
                            stored.knowledge_schema_version
                            != artifact.schema_version,
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
            store.put_paper(
                prepared,
                targets,
                vectors,
                embedding_model=self._embedding_model,
            )
            self._index = None

    async def put_structure_tree(self, tree: PaperStructureTree) -> None:
        """Persist one self-contained structure tree with no source or chunk set.

        Reads ``as_of`` / source ref / ``source_content_hash`` from the tree's
        own provenance metadata; storing it requires no source revision or chunk
        set. This is the dump half of the ``open_structure`` dump/load pair, and
        is what ``put(tree)`` dispatches to.
        """
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            prepared = store.prepare_structure_tree(tree)
            store.put_structure_tree(prepared)
            self._index = None

    async def put_paper_structure_tree(
        self,
        source: PaperSourceRevision,
        tree: PaperStructureTree,
    ) -> None:
        """Deprecated: persist a self-contained structure tree standalone.

        Retained for backward compatibility. The ``source`` argument is only
        checked for consistency with the tree; the tree is a self-contained
        artifact stored on its own (no source revision or chunk set). Prefer
        ``put(tree)`` / ``put_structure_tree(tree)``.
        """
        if tree.source_revision_id != source.id:
            raise ValueError("paper structure tree belongs to another source")
        await self.put_structure_tree(tree)

    async def get(self, item_id: UUID) -> BaseKnowledge:
        """Return validated canonical knowledge or report not-found/stale data."""
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            return store.get(item_id)

    async def get_paper(
        self,
        source_revision_id: UUID,
        *,
        chunk_set_id: UUID | None = None,
        summary_id: UUID | None = None,
    ) -> PaperSemanticResult:
        """Return one unambiguous source/chunk-set/summary combination."""
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            return store.get_paper_result(
                source_revision_id,
                chunk_set_id=chunk_set_id,
                summary_id=summary_id,
            )

    async def get_artifact(self, artifact_id: UUID) -> PaperArtifact:
        """Return one validated canonical paper artifact aggregate."""
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            return store.get_paper_artifact(artifact_id)

    async def open_structure(self, tree_id: UUID) -> PaperStructureTree:
        """Load one self-contained paper structure tree by its artifact id.

        The returned tree is a complete value: every leaf node carries its own
        text and provenance metadata, so callers can retrieve over it without
        any further library round-trip. This is the load counterpart of
        ``put(tree)`` / ``put_structure_tree``.
        """
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            artifact = store.get_paper_artifact(tree_id)
            if not isinstance(artifact, PaperStructureTree):
                raise KeyError(
                    f"Paper artifact '{tree_id}' is not a structure tree"
                )
            return artifact

    async def resolve(
        self, locator: ArtifactLocator
    ) -> BaseKnowledge | TreeNode | ResolvedPaperArtifact:
        """Resolve a semantic locator to its canonical aggregate or member."""
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            if locator.source_revision_id is not None:
                return store.resolve_paper_locator(locator)
            item = store.get(locator.artifact_id)
            if locator.artifact_kind != item.item_type:
                raise KeyError("Knowledge artifact locator metadata mismatch")
            if locator.member_id is None:
                return item
            if not isinstance(item, TreeKnowledge):
                raise KeyError("Flat knowledge artifacts do not have members")
            try:
                return item.nodes[locator.member_id]
            except KeyError as exc:
                raise KeyError(
                    f"Knowledge node '{locator.member_id}' not found"
                ) from exc

    async def search(self, query: SemanticQuery) -> list[SemanticHit]:
        """Rank filtered targets through the private LlamaIndex retriever."""
        async with self._lock:
            store = self._store
            if store is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            if self._index is None:
                self._index = _LlamaIndexRetriever(
                    store.load_index_records(
                        embedding_model=self._embedding_model,
                        embedding_dimensions=self._embedding_dimensions,
                    )
                )
            candidates = self._index.filter(query)
            if not candidates:
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
            ranked = self._index.rank(
                candidates, query_vectors[0], top_k=query.top_k
            )

            canonical: dict[UUID, BaseKnowledge] = {}
            paper_artifacts: dict[UUID, PaperArtifact] = {}
            paper_sources: dict[UUID, PaperSourceRevision] = {}
            hits: list[SemanticHit] = []
            for result in ranked:
                record = result.record
                locator = ArtifactLocator(
                    source_revision_id=record.source_revision_id,
                    artifact_id=record.item_id,
                    artifact_kind=record.artifact_kind,
                    member_id=record.node_id,
                )
                if record.owner_kind == "paper":
                    source_id = record.source_revision_id
                    if source_id is None:
                        raise RuntimeError(
                            f"Stale paper projection '{record.target_id}': "
                            "source locator is missing"
                        )
                    source = paper_sources.get(source_id)
                    if source is None:
                        source = store.get_paper_source(source_id)
                        paper_sources[source_id] = source
                    artifact = paper_artifacts.get(record.item_id)
                    if artifact is None:
                        artifact = store.get_paper_artifact(record.item_id)
                        paper_artifacts[record.item_id] = artifact
                    if record.node_id is None:
                        if not isinstance(artifact, PaperGlobalSummary):
                            raise RuntimeError(
                                f"Stale paper projection '{record.target_id}': "
                                "chunk-set aggregate is not searchable"
                            )
                        citations = [
                            Citation(
                                source_id=str(source_id),
                                page=citation.page_number,
                                quote=citation.quote,
                            )
                            for citation in artifact.citations
                        ]
                    elif isinstance(artifact, PaperChunkSet):
                        chunk = next(
                            (
                                value
                                for value in artifact.chunks
                                if value.chunk_id == record.node_id
                            ),
                            None,
                        )
                        if chunk is None:
                            raise RuntimeError(
                                f"Stale paper projection '{record.target_id}': "
                                "canonical chunk is missing"
                            )
                        citations = [
                            Citation(
                                source_id=str(source_id),
                                page=span.page_number,
                                char_offset=span.start_char,
                                quote=chunk.text[:500],
                            )
                            for span in chunk.source_spans
                        ]
                    else:
                        raise RuntimeError(
                            f"Stale paper projection '{record.target_id}': "
                            "summary unexpectedly has a member"
                        )
                    as_of = source.as_of
                    available_at = source.available_at
                    source_ref = source.source
                    item_type = artifact.artifact_kind
                else:
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
                                    f"'{record.target_id}': canonical node is "
                                    "missing"
                                )
                            citations = node.citations
                    else:
                        if record.node_id is not None:
                            raise RuntimeError(
                                f"Stale index data for target "
                                f"'{record.target_id}': node target belongs to "
                                "non-tree knowledge"
                            )
                        citations = item.citations
                    as_of = item.as_of
                    available_at = item.available_at
                    source_ref = item.source
                    item_type = item.item_type
                hits.append(
                    SemanticHit(
                        locator=locator,
                        projection=SearchProjection(
                            version=record.projection_version,
                            model=record.embedding_model,
                            dimensions=record.dimension,
                            content_hash=record.projection_hash,
                        ),
                        item_id=record.item_id,
                        node_id=record.node_id,
                        item_type=item_type,
                        score=result.score,
                        matched_text=record.matched_text,
                        as_of=as_of,
                        available_at=available_at,
                        source=source_ref,
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
