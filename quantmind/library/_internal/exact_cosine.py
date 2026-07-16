"""Deterministic NumPy exact-cosine filtering and ranking."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

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
    """One exact-cosine result before canonical hit resolution."""

    record: _IndexRecord
    score: float


def _coerce_provider_vectors(
    values: Any,
    *,
    expected_count: int,
    expected_dimensions: int | None,
) -> NDArray[np.float32]:
    """Validate provider output before it can enter the durable index."""
    try:
        vectors = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Embedding provider returned non-rectangular data"
        ) from exc
    if vectors.ndim != 2 or vectors.shape[0] != expected_count:
        raise ValueError(
            "Embedding provider returned an unexpected number or shape of vectors"
        )
    dimensions = int(vectors.shape[1])
    if dimensions < 1:
        raise ValueError("Embedding provider returned zero-dimensional vectors")
    if expected_dimensions is not None and dimensions != expected_dimensions:
        raise ValueError(
            "Embedding dimension mismatch: provider returned "
            f"{dimensions}, expected {expected_dimensions}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding provider returned non-finite values")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Embedding provider returned a zero vector")
    return np.ascontiguousarray(vectors, dtype=np.float32)


def _decode_stored_vector(
    blob: bytes, dimension: int, target_id: str
) -> NDArray[np.float32]:
    """Decode and validate a persisted vector as corrupt-index protection."""
    if dimension < 1 or len(blob) != dimension * np.dtype("<f4").itemsize:
        raise RuntimeError(
            f"Corrupt index data for target '{target_id}': "
            "stored byte length does not match its dimension"
        )
    vector = np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
    if not np.isfinite(vector).all():
        raise RuntimeError(
            f"Corrupt index data for target '{target_id}': non-finite vector"
        )
    if np.linalg.norm(vector) == 0:
        raise RuntimeError(
            f"Corrupt index data for target '{target_id}': zero vector"
        )
    return vector


def _encode_vector(vector: NDArray[np.float32]) -> tuple[bytes, int]:
    """Encode one validated provider vector for SQLite persistence."""
    return (
        np.asarray(vector, dtype="<f4").tobytes(),
        int(vector.shape[0]),
    )


def _timestamp(value: datetime, field_name: str) -> float:
    """Normalize an aware financial timestamp for deterministic filtering."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc).timestamp()


class _CosineCandidates:
    """Filtered records and normalized vectors ready for one query."""

    def __init__(
        self,
        records: tuple[_IndexRecord, ...],
        matrix: NDArray[np.float32],
    ) -> None:
        self._records = records
        self._matrix = matrix

    def rank(
        self,
        query_vector: NDArray[np.float32],
        *,
        top_k: int,
    ) -> list[_RankedRecord]:
        """Rank candidates best-first with stable target-ID tie breaking."""
        if query_vector.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch: query vector has "
                f"{query_vector.shape[0]} dimensions but the index has "
                f"{self._matrix.shape[1]}"
            )
        normalized = query_vector / np.linalg.norm(query_vector)
        scores = self._matrix @ normalized
        ranked = sorted(
            zip(self._records, scores, strict=True),
            key=lambda pair: (-float(pair[1]), pair[0].target_id),
        )[:top_k]
        return [
            _RankedRecord(record=record, score=float(score))
            for record, score in ranked
        ]


class _ExactCosineIndex:
    """Immutable in-memory exact-cosine index rebuilt from SQLite rows."""

    def __init__(
        self,
        records: tuple[_IndexRecord, ...],
        matrix: NDArray[np.float32],
    ) -> None:
        self._records = records
        self._matrix = matrix

    @classmethod
    def build(cls, records: Sequence[_IndexRecord]) -> "_ExactCosineIndex":
        """Validate durable vectors and build a normalized matrix."""
        vectors: list[NDArray[np.float32]] = []
        index_dimension: int | None = None
        for record in records:
            if index_dimension is None:
                index_dimension = record.dimension
            elif record.dimension != index_dimension:
                raise RuntimeError(
                    "Corrupt index data: stored targets have inconsistent "
                    "embedding dimensions"
                )
            vector = _decode_stored_vector(
                record.embedding,
                record.dimension,
                record.target_id,
            )
            vectors.append(
                np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)
            )
        matrix = (
            np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
            if vectors
            else np.empty((0, 0), dtype=np.float32)
        )
        return cls(tuple(records), matrix)

    def filter(self, query: SemanticQuery) -> _CosineCandidates | None:
        """Apply metadata and financial-time filters before query embedding."""
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
        indices: list[int] = []
        for index, record in enumerate(self._records):
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
            indices.append(index)
        if not indices:
            return None
        return _CosineCandidates(
            tuple(self._records[index] for index in indices),
            np.ascontiguousarray(self._matrix[indices], dtype=np.float32),
        )
