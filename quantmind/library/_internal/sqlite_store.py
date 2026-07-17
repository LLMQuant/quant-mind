"""Concrete SQLite persistence for canonical knowledge and index records."""

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from quantmind.knowledge import (
    BaseKnowledge,
    Earnings,
    Factor,
    News,
    Paper,
    PaperKnowledgeCard,
    Thesis,
    TreeKnowledge,
)
from quantmind.library._internal.llamaindex_retriever import _IndexRecord
from quantmind.library._internal.retrieval_targets import (
    _PROJECTION_SCHEMA_VERSION,
    _RetrievalTarget,
)

_DATABASE_SCHEMA_VERSION = 2

_KNOWLEDGE_CLASSES: dict[str, type[BaseKnowledge]] = {
    f"{knowledge_type.__module__}:{knowledge_type.__qualname__}": knowledge_type
    for knowledge_type in (
        Earnings,
        Factor,
        News,
        Paper,
        PaperKnowledgeCard,
        Thesis,
    )
}


@dataclass(frozen=True)
class _CanonicalNode:
    """One normalized canonical tree node."""

    node_id: UUID
    parent_id: UUID | None
    position: int
    payload: str
    content_hash: str


@dataclass(frozen=True)
class _CanonicalDocument:
    """Canonical aggregate root plus separately persisted tree nodes."""

    knowledge_class: str
    item_shape: str
    payload: str
    canonical_hash: str
    nodes: tuple[_CanonicalNode, ...]


@dataclass(frozen=True)
class _StoredEmbedding:
    """Existing vector and invalidation metadata for one target."""

    target_id: str
    embedding_model: str
    dimension: int
    projection_hash: str
    source_content_hash: str | None
    knowledge_schema_version: str
    projection_schema_version: str
    embedding: bytes


@dataclass(frozen=True)
class _PreparedPut:
    """Validated canonical write plus the vectors it may retain."""

    item: BaseKnowledge
    canonical: _CanonicalDocument
    as_of: float
    available_at: float | None
    tags_json: str
    existing_embeddings: dict[str, _StoredEmbedding]


def _json_payload(value: object) -> str:
    """Encode canonical JSON with stable ordering and separators."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_payload(item: BaseKnowledge) -> _CanonicalDocument:
    """Split a supported canonical aggregate into root and node records."""
    knowledge_class = f"{type(item).__module__}:{type(item).__qualname__}"
    if knowledge_class not in _KNOWLEDGE_CLASSES:
        raise TypeError(
            f"Unsupported knowledge type '{type(item).__name__}'; "
            "only canonical quantmind.knowledge types can be persisted"
        )
    full_payload = _json_payload(item.model_dump(mode="json"))
    canonical_hash = hashlib.sha256(full_payload.encode("utf-8")).hexdigest()
    if not isinstance(item, TreeKnowledge):
        return _CanonicalDocument(
            knowledge_class=knowledge_class,
            item_shape="flat",
            payload=full_payload,
            canonical_hash=canonical_hash,
            nodes=(),
        )

    nodes: list[_CanonicalNode] = []
    for node_id, node in sorted(
        item.nodes.items(), key=lambda pair: str(pair[0])
    ):
        node_payload = _json_payload(node.model_dump(mode="json"))
        nodes.append(
            _CanonicalNode(
                node_id=node_id,
                parent_id=node.parent_id,
                position=node.position,
                payload=node_payload,
                content_hash=hashlib.sha256(
                    node_payload.encode("utf-8")
                ).hexdigest(),
            )
        )
    return _CanonicalDocument(
        knowledge_class=knowledge_class,
        item_shape="tree",
        payload=_json_payload(item.model_dump(mode="json", exclude={"nodes"})),
        canonical_hash=canonical_hash,
        nodes=tuple(nodes),
    )


def _assemble_canonical_payload(
    *,
    item_id: str,
    item_shape: str,
    item_payload: str,
    expected_node_count: int,
    node_records: Sequence[tuple[str, str | None, int, str, str]],
) -> str:
    """Rehydrate normalized tree nodes into the canonical Pydantic payload."""
    if item_shape == "flat":
        if expected_node_count or node_records:
            raise RuntimeError(
                f"Stale canonical knowledge for item '{item_id}': "
                "flat knowledge unexpectedly has tree nodes"
            )
        return item_payload
    if item_shape != "tree":
        raise RuntimeError(
            f"Stale canonical knowledge for item '{item_id}': "
            f"unsupported item shape '{item_shape}'"
        )
    if len(node_records) != expected_node_count:
        raise RuntimeError(
            f"Stale canonical knowledge for item '{item_id}': expected "
            f"{expected_node_count} canonical nodes, found {len(node_records)}"
        )
    try:
        root_payload = json.loads(item_payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Stale canonical knowledge for item '{item_id}': invalid root JSON"
        ) from exc
    if not isinstance(root_payload, dict):
        raise RuntimeError(
            f"Stale canonical knowledge for item '{item_id}': invalid root payload"
        )
    nodes: dict[str, object] = {}
    for node_id, parent_id, position, node_payload, node_hash in node_records:
        actual_hash = hashlib.sha256(node_payload.encode("utf-8")).hexdigest()
        if actual_hash != node_hash:
            raise RuntimeError(
                f"Stale canonical knowledge for item '{item_id}': "
                f"node '{node_id}' content hash mismatch"
            )
        try:
            parsed_node = json.loads(node_payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Stale canonical knowledge for item '{item_id}': "
                f"node '{node_id}' contains invalid JSON"
            ) from exc
        if (
            not isinstance(parsed_node, dict)
            or parsed_node.get("node_id") != node_id
            or parsed_node.get("parent_id") != parent_id
            or parsed_node.get("position") != position
        ):
            raise RuntimeError(
                f"Stale canonical knowledge for item '{item_id}': "
                f"node '{node_id}' metadata does not match its payload"
            )
        nodes[node_id] = parsed_node
    root_payload["nodes"] = nodes
    return _json_payload(root_payload)


def _load_canonical(
    *,
    item_id: str,
    knowledge_class: str,
    item_type: str,
    schema_version: str,
    payload: str,
    canonical_hash: str,
) -> BaseKnowledge:
    """Validate stored canonical bytes against identity and schema metadata."""
    actual_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if actual_hash != canonical_hash:
        raise RuntimeError(
            f"Stale canonical knowledge for item '{item_id}': content hash mismatch"
        )
    model = _KNOWLEDGE_CLASSES.get(knowledge_class)
    if model is None:
        raise RuntimeError(
            f"Stale canonical knowledge for item '{item_id}': "
            f"unsupported stored type '{knowledge_class}'"
        )
    try:
        item = model.model_validate_json(payload)
    except ValidationError as exc:
        raise RuntimeError(
            f"Stale canonical knowledge for item '{item_id}': "
            "stored payload no longer validates"
        ) from exc
    if (
        str(item.id) != item_id
        or item.item_type != item_type
        or item.schema_version != schema_version
    ):
        raise RuntimeError(
            f"Stale canonical knowledge for item '{item_id}': identity or schema "
            "metadata does not match the payload"
        )
    return item


def _timestamp(value: datetime, field_name: str) -> float:
    """Normalize an aware canonical timestamp for SQLite filtering."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc).timestamp()


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


class _SQLiteStore:
    """Own the concrete SQLite schema, transactions, and record validation."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db

    @classmethod
    def open(cls, path: str | Path) -> "_SQLiteStore":
        """Open and initialize a concrete SQLite knowledge store."""
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
            db = sqlite3.connect(database_path, isolation_level=None)
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
        return cls(db)

    def prepare_put(self, item: BaseKnowledge) -> _PreparedPut:
        """Validate a canonical write and load vectors it may retain."""
        rows = self._db.execute(
            "SELECT * FROM semantic_records WHERE item_id = ?",
            (str(item.id),),
        ).fetchall()
        existing = {
            str(row["target_id"]): _StoredEmbedding(
                target_id=str(row["target_id"]),
                embedding_model=str(row["embedding_model"]),
                dimension=int(row["dimension"]),
                projection_hash=str(row["projection_hash"]),
                source_content_hash=(
                    str(row["source_content_hash"])
                    if row["source_content_hash"] is not None
                    else None
                ),
                knowledge_schema_version=str(row["knowledge_schema_version"]),
                projection_schema_version=str(row["projection_schema_version"]),
                embedding=bytes(row["embedding"]),
            )
            for row in rows
        }
        return _PreparedPut(
            item=item,
            canonical=_canonical_payload(item),
            as_of=_timestamp(item.as_of, "BaseKnowledge.as_of"),
            available_at=(
                _timestamp(item.available_at, "BaseKnowledge.available_at")
                if item.available_at is not None
                else None
            ),
            tags_json=json.dumps(
                item.tags,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            existing_embeddings=existing,
        )

    def put(
        self,
        prepared: _PreparedPut,
        targets: Sequence[_RetrievalTarget],
        vectors: dict[str, tuple[bytes, int]],
        *,
        embedding_model: str,
    ) -> None:
        """Atomically replace canonical and derived records for one item."""
        item = prepared.item
        canonical = prepared.canonical
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
                    len(targets),
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
            for target in targets:
                blob, dimension = vectors[target.target_id]
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
                        target.target_id,
                        str(item.id),
                        str(target.node_id)
                        if target.node_id is not None
                        else None,
                        item.item_type,
                        target.text,
                        prepared.as_of,
                        prepared.available_at,
                        item.source.kind,
                        item.confidence,
                        prepared.tags_json,
                        str(target.tree_id)
                        if target.tree_id is not None
                        else None,
                        embedding_model,
                        dimension,
                        target.projection_hash,
                        item.source.content_hash,
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

    def get(self, item_id: UUID) -> BaseKnowledge:
        """Return validated canonical knowledge or report not-found/stale data."""
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
        item_key = str(row["item_id"])
        node_rows = self._db.execute(
            """
            SELECT node_id, parent_id, position, payload_json, content_hash
            FROM knowledge_nodes
            WHERE item_id = ?
            ORDER BY node_id
            """,
            (item_key,),
        ).fetchall()
        payload = _assemble_canonical_payload(
            item_id=item_key,
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
            item_id=item_key,
            knowledge_class=str(row["knowledge_class"]),
            item_type=str(row["item_type"]),
            schema_version=str(row["schema_version"]),
            payload=payload,
            canonical_hash=str(row["canonical_hash"]),
        )

    def delete(self, item_id: UUID) -> None:
        """Transactionally remove canonical knowledge and every child record."""
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

    def load_index_records(
        self,
        *,
        embedding_model: str,
        embedding_dimensions: int | None,
    ) -> list[_IndexRecord]:
        """Validate SQLite relationships and load typed exact-index records."""
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
        for row in rows:
            target_id = str(row["target_id"])
            dimension = int(row["dimension"])
            if str(row["embedding_model"]) != embedding_model:
                raise RuntimeError(
                    f"Stale index data for target '{target_id}': embedding model "
                    "changed; re-put the canonical item"
                )
            if (
                embedding_dimensions is not None
                and dimension != embedding_dimensions
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
                    dimension=dimension,
                    embedding=bytes(row["embedding"]),
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Corrupt index data for target '{target_id}': invalid metadata"
                ) from exc
            records.append(record)
        return records

    def close(self) -> None:
        """Close the owned SQLite connection."""
        self._db.close()
