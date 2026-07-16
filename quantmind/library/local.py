"""Opinionated SQLite and NumPy semantic knowledge library."""

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from quantmind.knowledge import BaseKnowledge, TreeKnowledge
from quantmind.library._embed import _OpenAIEmbeddingProvider
from quantmind.library._ports import _EmbeddingProvider
from quantmind.library._projection import (
    _PROJECTION_SCHEMA_VERSION,
    _assemble_canonical_payload,
    _canonical_payload,
    _load_canonical,
    _project_knowledge,
    _Projection,
)
from quantmind.library._types import SemanticHit, SemanticQuery

_DATABASE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class _IndexRecord:
    """Validated metadata aligned with one NumPy matrix row."""

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


def _timestamp(value: datetime, field_name: str) -> float:
    """Normalize an aware financial timestamp for deterministic filtering."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc).timestamp()


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


def _initialize_schema(db: sqlite3.Connection) -> None:
    """Create the current schema or reject an incompatible local database."""
    version_row = db.execute("PRAGMA user_version").fetchone()
    version = int(version_row[0])
    if version not in (0, _DATABASE_SCHEMA_VERSION):
        raise RuntimeError(
            "Stale knowledge library schema: database version "
            f"{version}, expected {_DATABASE_SCHEMA_VERSION}"
        )
    if version == _DATABASE_SCHEMA_VERSION:
        return
    db.executescript(
        f"""
        CREATE TABLE knowledge_items (
            item_id TEXT PRIMARY KEY,
            knowledge_class TEXT NOT NULL,
            item_type TEXT NOT NULL,
            item_shape TEXT NOT NULL CHECK (item_shape IN ('flat', 'tree')),
            schema_version TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            canonical_hash TEXT NOT NULL,
            node_count INTEGER NOT NULL CHECK (node_count >= 0),
            target_count INTEGER NOT NULL CHECK (target_count > 0)
        );

        CREATE TABLE knowledge_nodes (
            item_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            parent_id TEXT,
            position INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            PRIMARY KEY (item_id, node_id),
            FOREIGN KEY (item_id) REFERENCES knowledge_items(item_id)
                ON DELETE CASCADE
        );

        CREATE TABLE semantic_records (
            target_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            node_id TEXT,
            item_type TEXT NOT NULL,
            matched_text TEXT NOT NULL,
            as_of REAL NOT NULL,
            available_at REAL,
            source_kind TEXT NOT NULL,
            confidence TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            tree_id TEXT,
            embedding_model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            projection_hash TEXT NOT NULL,
            source_content_hash TEXT,
            knowledge_schema_version TEXT NOT NULL,
            projection_schema_version TEXT NOT NULL,
            item_canonical_hash TEXT NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY (item_id) REFERENCES knowledge_items(item_id)
                ON DELETE CASCADE
        );

        CREATE INDEX semantic_records_item_id
            ON semantic_records(item_id);
        CREATE INDEX semantic_records_filters
            ON semantic_records(item_type, source_kind, confidence, tree_id);
        CREATE INDEX knowledge_nodes_parent
            ON knowledge_nodes(item_id, parent_id, position);

        PRAGMA user_version = {_DATABASE_SCHEMA_VERSION};
        """
    )


def _load_stored_canonical(
    db: sqlite3.Connection, row: sqlite3.Row
) -> BaseKnowledge:
    """Rehydrate and validate one canonical aggregate from normalized rows."""
    item_id = str(row["item_id"])
    node_rows = db.execute(
        """
        SELECT node_id, parent_id, position, payload_json, content_hash
        FROM knowledge_nodes
        WHERE item_id = ?
        ORDER BY node_id
        """,
        (item_id,),
    ).fetchall()
    payload = _assemble_canonical_payload(
        item_id=item_id,
        item_shape=str(row["item_shape"]),
        item_payload=str(row["payload_json"]),
        expected_node_count=int(row["node_count"]),
        node_records=[
            (
                str(node_row["node_id"]),
                (
                    str(node_row["parent_id"])
                    if node_row["parent_id"] is not None
                    else None
                ),
                int(node_row["position"]),
                str(node_row["payload_json"]),
                str(node_row["content_hash"]),
            )
            for node_row in node_rows
        ],
    )
    return _load_canonical(
        item_id=item_id,
        knowledge_class=str(row["knowledge_class"]),
        item_type=str(row["item_type"]),
        schema_version=str(row["schema_version"]),
        payload=payload,
        canonical_hash=str(row["canonical_hash"]),
    )


class LocalKnowledgeLibrary:
    """Persist and semantically search canonical QuantMind knowledge locally."""

    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        embedding_model: str,
        embedding_dimensions: int | None,
        embedding_provider: _EmbeddingProvider,
    ) -> None:
        self._db: sqlite3.Connection | None = db
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._embedding_provider = embedding_provider
        self._lock = asyncio.Lock()
        self._index_records: list[_IndexRecord] | None = None
        self._index_matrix: NDArray[np.float32] | None = None

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
        supplied_path = str(path)
        database_path = (
            supplied_path
            if supplied_path == ":memory:"
            else str(Path(supplied_path).expanduser())
        )
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        db: sqlite3.Connection | None = None
        try:
            db = sqlite3.connect(
                database_path,
                isolation_level=None,
            )
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA busy_timeout = 5000")
            _initialize_schema(db)
        except sqlite3.DatabaseError as exc:
            if db is not None:
                db.close()
            raise RuntimeError(
                f"Corrupt knowledge library database at '{database_path}'"
            ) from exc
        except Exception:
            if db is not None:
                db.close()
            raise
        provider = _embedding_provider or _OpenAIEmbeddingProvider()
        return cls(
            db,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            embedding_provider=provider,
        )

    async def put(self, item: BaseKnowledge) -> None:
        """Atomically persist canonical knowledge and its affected projections."""
        async with self._lock:
            if self._db is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            canonical = _canonical_payload(item)
            projections = _project_knowledge(item)
            as_of = _timestamp(item.as_of, "BaseKnowledge.as_of")
            available_at = (
                _timestamp(item.available_at, "BaseKnowledge.available_at")
                if item.available_at is not None
                else None
            )
            tags_json = json.dumps(
                item.tags,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            existing_rows = self._db.execute(
                "SELECT * FROM semantic_records WHERE item_id = ?",
                (str(item.id),),
            ).fetchall()
            existing = {str(row["target_id"]): row for row in existing_rows}
            source_content_hash = item.source.content_hash

            affected: list[_Projection] = []
            retained: dict[str, tuple[bytes, int]] = {}
            for projection in projections:
                row = existing.get(projection.target_id)
                needs_embedding = row is None
                if row is not None:
                    needs_embedding = any(
                        (
                            str(row["embedding_model"])
                            != self._embedding_model,
                            str(row["projection_hash"])
                            != projection.projection_hash,
                            row["source_content_hash"] != source_content_hash,
                            str(row["knowledge_schema_version"])
                            != item.schema_version,
                            str(row["projection_schema_version"])
                            != _PROJECTION_SCHEMA_VERSION,
                            self._embedding_dimensions is not None
                            and int(row["dimension"])
                            != self._embedding_dimensions,
                        )
                    )
                if needs_embedding:
                    affected.append(projection)
                else:
                    assert row is not None
                    blob = bytes(row["embedding"])
                    dimension = int(row["dimension"])
                    _decode_stored_vector(
                        blob,
                        dimension,
                        projection.target_id,
                    )
                    retained[projection.target_id] = (blob, dimension)

            generated: dict[str, tuple[bytes, int]] = {}
            if affected:
                provider_values = await self._embedding_provider.embed(
                    [projection.text for projection in affected],
                    model=self._embedding_model,
                    dimensions=self._embedding_dimensions,
                )
                vectors = _coerce_provider_vectors(
                    provider_values,
                    expected_count=len(affected),
                    expected_dimensions=self._embedding_dimensions,
                )
                for projection, vector in zip(affected, vectors, strict=True):
                    stored = np.asarray(vector, dtype="<f4").tobytes()
                    generated[projection.target_id] = (
                        stored,
                        int(vector.shape[0]),
                    )

            try:
                self._db.execute("BEGIN IMMEDIATE")
                self._db.execute(
                    """
                    INSERT INTO knowledge_items (
                        item_id, knowledge_class, item_type, item_shape,
                        schema_version, payload_json, canonical_hash,
                        node_count, target_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(item_id) DO UPDATE SET
                        knowledge_class = excluded.knowledge_class,
                        item_type = excluded.item_type,
                        item_shape = excluded.item_shape,
                        schema_version = excluded.schema_version,
                        payload_json = excluded.payload_json,
                        canonical_hash = excluded.canonical_hash,
                        node_count = excluded.node_count,
                        target_count = excluded.target_count
                    """,
                    (
                        str(item.id),
                        canonical.knowledge_class,
                        item.item_type,
                        canonical.item_shape,
                        item.schema_version,
                        canonical.payload,
                        canonical.canonical_hash,
                        len(canonical.nodes),
                        len(projections),
                    ),
                )
                self._db.execute(
                    "DELETE FROM semantic_records WHERE item_id = ?",
                    (str(item.id),),
                )
                self._db.execute(
                    "DELETE FROM knowledge_nodes WHERE item_id = ?",
                    (str(item.id),),
                )
                for node in canonical.nodes:
                    self._db.execute(
                        """
                        INSERT INTO knowledge_nodes (
                            item_id, node_id, parent_id, position,
                            payload_json, content_hash
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(item.id),
                            str(node.node_id),
                            str(node.parent_id) if node.parent_id else None,
                            node.position,
                            node.payload,
                            node.content_hash,
                        ),
                    )
                for projection in projections:
                    stored_vector = generated.get(projection.target_id)
                    if stored_vector is None:
                        stored_vector = retained[projection.target_id]
                    blob, dimension = stored_vector
                    self._db.execute(
                        """
                        INSERT INTO semantic_records (
                            target_id, item_id, node_id, item_type,
                            matched_text, as_of, available_at, source_kind,
                            confidence, tags_json, tree_id, embedding_model,
                            dimension, projection_hash, source_content_hash,
                            knowledge_schema_version,
                            projection_schema_version, item_canonical_hash,
                            embedding
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?
                        )
                        """,
                        (
                            projection.target_id,
                            str(item.id),
                            (
                                str(projection.node_id)
                                if projection.node_id is not None
                                else None
                            ),
                            item.item_type,
                            projection.text,
                            as_of,
                            available_at,
                            item.source.kind,
                            item.confidence,
                            tags_json,
                            (
                                str(projection.tree_id)
                                if projection.tree_id is not None
                                else None
                            ),
                            self._embedding_model,
                            dimension,
                            projection.projection_hash,
                            source_content_hash,
                            item.schema_version,
                            _PROJECTION_SCHEMA_VERSION,
                            canonical.canonical_hash,
                            blob,
                        ),
                    )
                self._db.execute("COMMIT")
            except Exception:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise
            self._index_records = None
            self._index_matrix = None

    async def get(self, item_id: UUID) -> BaseKnowledge:
        """Return validated canonical knowledge or report not-found/stale data."""
        async with self._lock:
            if self._db is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            row = self._db.execute(
                "SELECT * FROM knowledge_items WHERE item_id = ?",
                (str(item_id),),
            ).fetchone()
            if row is None:
                derived_count = int(
                    self._db.execute(
                        """
                        SELECT
                            (SELECT COUNT(*) FROM semantic_records
                             WHERE item_id = ?)
                            +
                            (SELECT COUNT(*) FROM knowledge_nodes
                             WHERE item_id = ?)
                        """,
                        (str(item_id), str(item_id)),
                    ).fetchone()[0]
                )
                if derived_count:
                    raise RuntimeError(
                        f"Stale data for item '{item_id}': child records exist "
                        "without canonical knowledge"
                    )
                raise KeyError(f"Knowledge item '{item_id}' not found")
            return _load_stored_canonical(self._db, row)

    async def search(self, query: SemanticQuery) -> list[SemanticHit]:
        """Rank filtered projections with deterministic exact cosine similarity."""
        async with self._lock:
            if self._db is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            if self._index_records is None or self._index_matrix is None:
                self._rebuild_numpy_index()
            assert self._index_records is not None
            assert self._index_matrix is not None
            index_records = self._index_records
            index_matrix = self._index_matrix

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
            source_kinds = (
                set(query.source_kinds) if query.source_kinds else None
            )
            required_tags = set(query.tags) if query.tags else None
            candidates: list[int] = []
            for index, record in enumerate(index_records):
                if (
                    item_types is not None
                    and record.item_type not in item_types
                ):
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
                if (
                    query.tree_id is not None
                    and record.tree_id != query.tree_id
                ):
                    continue
                if as_of_before is not None and record.as_of > as_of_before:
                    continue
                if available_at_before is not None and (
                    record.available_at is None
                    or record.available_at > available_at_before
                ):
                    continue
                candidates.append(index)

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
            query_vector = query_vectors[0]
            if query_vector.shape[0] != index_matrix.shape[1]:
                raise ValueError(
                    "Embedding dimension mismatch: query vector has "
                    f"{query_vector.shape[0]} dimensions but the index has "
                    f"{index_matrix.shape[1]}"
                )
            query_vector = query_vector / np.linalg.norm(query_vector)
            scores = index_matrix[candidates] @ query_vector
            ranked = sorted(
                zip(candidates, scores, strict=True),
                key=lambda pair: (
                    -float(pair[1]),
                    index_records[pair[0]].target_id,
                ),
            )[: query.top_k]

            canonical: dict[UUID, BaseKnowledge] = {}
            hits: list[SemanticHit] = []
            for index, score in ranked:
                record = index_records[index]
                item = canonical.get(record.item_id)
                if item is None:
                    row = self._db.execute(
                        "SELECT * FROM knowledge_items WHERE item_id = ?",
                        (str(record.item_id),),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError(
                            f"Stale index data for item '{record.item_id}': "
                            "canonical knowledge is missing"
                        )
                    item = _load_stored_canonical(self._db, row)
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
                        score=float(score),
                        matched_text=record.matched_text,
                        as_of=item.as_of,
                        available_at=item.available_at,
                        source=item.source,
                        citations=citations,
                    )
                )
            return hits

    async def delete(self, item_id: UUID) -> None:
        """Transactionally remove canonical knowledge and every derived target."""
        async with self._lock:
            if self._db is None:
                raise RuntimeError("LocalKnowledgeLibrary is closed")
            exists = self._db.execute(
                "SELECT 1 FROM knowledge_items WHERE item_id = ?",
                (str(item_id),),
            ).fetchone()
            if exists is None:
                derived_count = int(
                    self._db.execute(
                        """
                        SELECT
                            (SELECT COUNT(*) FROM semantic_records
                             WHERE item_id = ?)
                            +
                            (SELECT COUNT(*) FROM knowledge_nodes
                             WHERE item_id = ?)
                        """,
                        (str(item_id), str(item_id)),
                    ).fetchone()[0]
                )
                if derived_count:
                    raise RuntimeError(
                        f"Stale data for item '{item_id}': cannot delete child "
                        "records without canonical knowledge"
                    )
                raise KeyError(f"Knowledge item '{item_id}' not found")
            try:
                self._db.execute("BEGIN IMMEDIATE")
                self._db.execute(
                    "DELETE FROM semantic_records WHERE item_id = ?",
                    (str(item_id),),
                )
                self._db.execute(
                    "DELETE FROM knowledge_nodes WHERE item_id = ?",
                    (str(item_id),),
                )
                self._db.execute(
                    "DELETE FROM knowledge_items WHERE item_id = ?",
                    (str(item_id),),
                )
                self._db.execute("COMMIT")
            except Exception:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise
            self._index_records = None
            self._index_matrix = None

    async def close(self) -> None:
        """Close SQLite and provider-owned resources; repeated calls are safe."""
        async with self._lock:
            if self._db is None:
                return
            db = self._db
            self._db = None
            self._index_records = None
            self._index_matrix = None
            try:
                await self._embedding_provider.close()
            finally:
                db.close()

    def _rebuild_numpy_index(self) -> None:
        """Validate durable records and rebuild the exact-cosine matrix."""
        assert self._db is not None
        orphan = self._db.execute(
            """
            SELECT r.item_id
            FROM semantic_records AS r
            LEFT JOIN knowledge_items AS i ON i.item_id = r.item_id
            WHERE i.item_id IS NULL
            LIMIT 1
            """
        ).fetchone()
        if orphan is not None:
            raise RuntimeError(
                f"Stale index data for item '{orphan['item_id']}': "
                "derived records exist without canonical knowledge"
            )
        orphan_node = self._db.execute(
            """
            SELECT n.item_id
            FROM knowledge_nodes AS n
            LEFT JOIN knowledge_items AS i ON i.item_id = n.item_id
            WHERE i.item_id IS NULL
            LIMIT 1
            """
        ).fetchone()
        if orphan_node is not None:
            raise RuntimeError(
                f"Stale canonical knowledge for item '{orphan_node['item_id']}': "
                "tree nodes exist without an aggregate root"
            )
        incomplete_tree = self._db.execute(
            """
            SELECT i.item_id, i.item_shape, i.node_count,
                   COUNT(n.node_id) AS actual_count
            FROM knowledge_items AS i
            LEFT JOIN knowledge_nodes AS n ON n.item_id = i.item_id
            GROUP BY i.item_id, i.item_shape, i.node_count
            HAVING COUNT(n.node_id) != i.node_count
                OR (i.item_shape = 'flat' AND i.node_count != 0)
                OR (i.item_shape = 'tree' AND i.node_count = 0)
            LIMIT 1
            """
        ).fetchone()
        if incomplete_tree is not None:
            raise RuntimeError(
                f"Stale canonical knowledge for item "
                f"'{incomplete_tree['item_id']}': expected "
                f"{incomplete_tree['node_count']} tree nodes, found "
                f"{incomplete_tree['actual_count']}"
            )
        incomplete = self._db.execute(
            """
            SELECT i.item_id, i.target_count, COUNT(r.target_id) AS actual_count
            FROM knowledge_items AS i
            LEFT JOIN semantic_records AS r ON r.item_id = i.item_id
            GROUP BY i.item_id, i.target_count
            HAVING COUNT(r.target_id) != i.target_count
            LIMIT 1
            """
        ).fetchone()
        if incomplete is not None:
            raise RuntimeError(
                f"Stale index data for item '{incomplete['item_id']}': expected "
                f"{incomplete['target_count']} targets, found "
                f"{incomplete['actual_count']}"
            )

        rows = self._db.execute(
            """
            SELECT r.*, i.canonical_hash AS current_canonical_hash,
                   i.schema_version AS current_schema_version
            FROM semantic_records AS r
            JOIN knowledge_items AS i ON i.item_id = r.item_id
            ORDER BY r.target_id
            """
        ).fetchall()
        records: list[_IndexRecord] = []
        vectors: list[NDArray[np.float32]] = []
        index_dimension: int | None = None
        for row in rows:
            target_id = str(row["target_id"])
            dimension = int(row["dimension"])
            if str(row["embedding_model"]) != self._embedding_model:
                raise RuntimeError(
                    f"Stale index data for target '{target_id}': embedding model "
                    "changed; re-put the canonical item"
                )
            if (
                self._embedding_dimensions is not None
                and dimension != self._embedding_dimensions
            ):
                raise RuntimeError(
                    f"Stale index data for target '{target_id}': embedding "
                    "dimension changed; re-put the canonical item"
                )
            if (
                str(row["projection_schema_version"])
                != _PROJECTION_SCHEMA_VERSION
                or str(row["knowledge_schema_version"])
                != str(row["current_schema_version"])
                or str(row["item_canonical_hash"])
                != str(row["current_canonical_hash"])
            ):
                raise RuntimeError(
                    f"Stale index data for target '{target_id}': projection or "
                    "canonical metadata changed; re-put the canonical item"
                )
            if index_dimension is None:
                index_dimension = dimension
            elif dimension != index_dimension:
                raise RuntimeError(
                    "Corrupt index data: stored targets have inconsistent "
                    "embedding dimensions"
                )
            vector = _decode_stored_vector(
                bytes(row["embedding"]), dimension, target_id
            )
            try:
                parsed_tags = json.loads(str(row["tags_json"]))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Corrupt index data for target '{target_id}': invalid tags"
                ) from exc
            if not isinstance(parsed_tags, list) or not all(
                isinstance(tag, str) for tag in parsed_tags
            ):
                raise RuntimeError(
                    f"Corrupt index data for target '{target_id}': invalid tags"
                )
            node_value = row["node_id"]
            tree_value = row["tree_id"]
            try:
                record = _IndexRecord(
                    target_id=target_id,
                    item_id=UUID(str(row["item_id"])),
                    node_id=UUID(str(node_value)) if node_value else None,
                    item_type=str(row["item_type"]),
                    matched_text=str(row["matched_text"]),
                    as_of=float(row["as_of"]),
                    available_at=(
                        float(row["available_at"])
                        if row["available_at"] is not None
                        else None
                    ),
                    source_kind=str(row["source_kind"]),
                    confidence=str(row["confidence"]),
                    tags=frozenset(parsed_tags),
                    tree_id=UUID(str(tree_value)) if tree_value else None,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Corrupt index data for target '{target_id}': invalid metadata"
                ) from exc
            records.append(record)
            vectors.append(
                np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)
            )

        self._index_records = records
        if vectors:
            self._index_matrix = np.ascontiguousarray(
                np.vstack(vectors), dtype=np.float32
            )
        else:
            self._index_matrix = np.empty((0, 0), dtype=np.float32)
