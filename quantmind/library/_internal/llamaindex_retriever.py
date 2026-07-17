"""Private LlamaIndex vector retrieval over rebuildable SQLite records."""

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import QueryBundle, TextNode

from quantmind.library._types import SemanticQuery


@dataclass(frozen=True)
class _IndexRecord:
    """Durable metadata and vector bytes for one searchable target."""

    target_id: str
    item_id: UUID
    node_id: UUID | None
    item_type: str
    matched_text: str
    as_of: float
    available_at: float | None
    source_kind: str
    confidence: str
    tags: frozenset[str]
    tree_id: UUID | None
    dimension: int
    embedding: bytes


@dataclass(frozen=True)
class _RankedRecord:
    """One LlamaIndex result before canonical hit resolution."""

    record: _IndexRecord
    score: float


def _coerce_provider_vectors(
    values: Any,
    *,
    expected_count: int,
    expected_dimensions: int | None,
) -> list[list[float]]:
    """Validate provider output before it can enter durable derived state."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Embedding provider returned non-rectangular data")
    if len(values) != expected_count:
        raise ValueError(
            "Embedding provider returned an unexpected number or shape of vectors"
        )
    vectors: list[list[float]] = []
    dimensions: int | None = None
    for value in values:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("Embedding provider returned non-rectangular data")
        try:
            vector = [float(component) for component in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Embedding provider returned non-rectangular data"
            ) from exc
        if dimensions is None:
            dimensions = len(vector)
        elif len(vector) != dimensions:
            raise ValueError("Embedding provider returned non-rectangular data")
        if not vector:
            raise ValueError(
                "Embedding provider returned zero-dimensional vectors"
            )
        if (
            expected_dimensions is not None
            and len(vector) != expected_dimensions
        ):
            raise ValueError(
                "Embedding dimension mismatch: provider returned "
                f"{len(vector)}, expected {expected_dimensions}"
            )
        if not all(math.isfinite(component) for component in vector):
            raise ValueError("Embedding provider returned non-finite values")
        if math.sqrt(sum(component * component for component in vector)) == 0:
            raise ValueError("Embedding provider returned a zero vector")
        vectors.append(vector)
    return vectors


def _decode_stored_vector(
    blob: bytes, dimension: int, target_id: str
) -> list[float]:
    """Decode and validate a persisted little-endian float vector."""
    if dimension < 1 or len(blob) != dimension * 4:
        raise RuntimeError(
            f"Corrupt index data for target '{target_id}': "
            "stored byte length does not match its dimension"
        )
    vector = list(struct.unpack(f"<{dimension}f", blob))
    if not all(math.isfinite(component) for component in vector):
        raise RuntimeError(
            f"Corrupt index data for target '{target_id}': non-finite vector"
        )
    if math.sqrt(sum(component * component for component in vector)) == 0:
        raise RuntimeError(
            f"Corrupt index data for target '{target_id}': zero vector"
        )
    return vector


def _encode_vector(vector: Sequence[float]) -> tuple[bytes, int]:
    """Encode one validated provider vector for SQLite persistence."""
    return struct.pack(f"<{len(vector)}f", *vector), len(vector)


def _timestamp(value: datetime, field_name: str) -> float:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc).timestamp()


class _LlamaIndexRetriever:
    """Filter durable records and rank them through LlamaIndex."""

    def __init__(self, records: Sequence[_IndexRecord]) -> None:
        dimensions = {record.dimension for record in records}
        if len(dimensions) > 1:
            raise RuntimeError(
                "Corrupt index data: stored targets have inconsistent "
                "embedding dimensions"
            )
        for record in records:
            _decode_stored_vector(
                record.embedding, record.dimension, record.target_id
            )
        self._records = tuple(records)

    def filter(self, query: SemanticQuery) -> tuple[_IndexRecord, ...]:
        """Apply metadata and financial-time filters before ranking."""
        as_of_before = (
            _timestamp(query.as_of_before, "SemanticQuery.as_of_before")
            if query.as_of_before is not None
            else None
        )
        available_at_before = (
            _timestamp(
                query.available_at_before,
                "SemanticQuery.available_at_before",
            )
            if query.available_at_before is not None
            else None
        )
        item_types = set(query.item_types) if query.item_types else None
        source_kinds = set(query.source_kinds) if query.source_kinds else None
        required_tags = set(query.tags) if query.tags else None
        selected: list[_IndexRecord] = []
        for record in self._records:
            if item_types is not None and record.item_type not in item_types:
                continue
            if (
                source_kinds is not None
                and record.source_kind not in source_kinds
            ):
                continue
            if (
                query.confidence is not None
                and record.confidence != query.confidence
            ):
                continue
            if required_tags is not None and not required_tags.issubset(
                record.tags
            ):
                continue
            if query.tree_id is not None and record.tree_id != query.tree_id:
                continue
            if as_of_before is not None and record.as_of > as_of_before:
                continue
            if available_at_before is not None and (
                record.available_at is None
                or record.available_at > available_at_before
            ):
                continue
            selected.append(record)
        return tuple(selected)

    @staticmethod
    def rank(
        records: tuple[_IndexRecord, ...],
        query_vector: Sequence[float],
        *,
        top_k: int,
    ) -> list[_RankedRecord]:
        """Rank filtered records through a private LlamaIndex vector index."""
        if not records:
            return []
        dimension = records[0].dimension
        if len(query_vector) != dimension:
            raise ValueError(
                "Embedding dimension mismatch: query vector has "
                f"{len(query_vector)} dimensions but the index has {dimension}"
            )
        nodes = [
            TextNode(
                id_=record.target_id,
                text=record.matched_text,
                embedding=_decode_stored_vector(
                    record.embedding, record.dimension, record.target_id
                ),
            )
            for record in records
        ]
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=MockEmbedding(embed_dim=dimension),
        )
        retriever = index.as_retriever(
            similarity_top_k=min(top_k, len(records))
        )
        by_target = {record.target_id: record for record in records}
        ranked = [
            _RankedRecord(
                record=by_target[result.node_id],
                score=float(result.score or 0.0),
            )
            for result in retriever.retrieve(
                QueryBundle(query_str="", embedding=list(query_vector))
            )
        ]
        return sorted(
            ranked,
            key=lambda result: (-result.score, result.record.target_id),
        )
