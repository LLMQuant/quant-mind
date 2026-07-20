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
    ArtifactLocator,
    BaseKnowledge,
    Citation,
    Earnings,
    Factor,
    LegacyPaper,
    News,
    PaperArtifact,
    PaperChunkSet,
    PaperFlowResult,
    PaperGlobalSummary,
    PaperSourceRevision,
    PaperStructureTree,
    ResolvedPaperArtifact,
    Thesis,
    TreeKnowledge,
)
from quantmind.knowledge.paper import (
    _validate_chunk_set_source,
    _validate_structure_tree_source,
)
from quantmind.library._internal.llamaindex_retriever import _IndexRecord
from quantmind.library._internal.retrieval_targets import (
    _PROJECTION_SCHEMA_VERSION,
    _RetrievalTarget,
)

_DATABASE_SCHEMA_VERSION = 4

_KNOWLEDGE_CLASSES: dict[str, type[BaseKnowledge]] = {
    f"{knowledge_type.__module__}:{knowledge_type.__qualname__}": knowledge_type
    for knowledge_type in (
        Earnings,
        Factor,
        News,
        LegacyPaper,
        Thesis,
    )
}
_KNOWLEDGE_CLASSES["quantmind.knowledge.paper:Paper"] = LegacyPaper


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


@dataclass(frozen=True)
class _CanonicalPaperMember:
    """One separately addressable paper-artifact member."""

    member_id: UUID
    parent_id: UUID | None
    position: int
    payload: str
    content_hash: str


@dataclass(frozen=True)
class _CanonicalPaperArtifact:
    """One aggregate artifact plus normalized member rows."""

    artifact: PaperArtifact
    payload: str
    canonical_hash: str
    members: tuple[_CanonicalPaperMember, ...]


@dataclass(frozen=True)
class _PreparedPaperPut:
    """Validated source/artifact write plus reusable search projections."""

    result: PaperFlowResult
    source_payload: str
    source_canonical_hash: str
    artifacts: tuple[_CanonicalPaperArtifact, ...]
    existing_embeddings: dict[str, _StoredEmbedding]


@dataclass(frozen=True)
class _PreparedPaperStructurePut:
    """Validated vectorless structure-tree write."""

    source: PaperSourceRevision
    source_payload: str
    source_canonical_hash: str
    tree: PaperStructureTree
    canonical: _CanonicalPaperArtifact


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


def _canonical_paper_artifact(
    artifact: PaperArtifact,
) -> _CanonicalPaperArtifact:
    """Normalize a paper artifact without losing its aggregate hash."""
    full_payload = _json_payload(artifact.model_dump(mode="json"))
    canonical_hash = hashlib.sha256(full_payload.encode("utf-8")).hexdigest()
    if isinstance(artifact, PaperGlobalSummary):
        return _CanonicalPaperArtifact(
            artifact=artifact,
            payload=full_payload,
            canonical_hash=canonical_hash,
            members=(),
        )
    if isinstance(artifact, PaperStructureTree):
        members = tuple(
            _CanonicalPaperMember(
                member_id=node.node_id,
                parent_id=node.parent_id,
                position=node.position,
                payload=(
                    payload := _json_payload(node.model_dump(mode="json"))
                ),
                content_hash=hashlib.sha256(
                    payload.encode("utf-8")
                ).hexdigest(),
            )
            for _, node in sorted(
                artifact.nodes.items(), key=lambda pair: str(pair[0])
            )
        )
        return _CanonicalPaperArtifact(
            artifact=artifact,
            payload=_json_payload(
                artifact.model_dump(mode="json", exclude={"nodes"})
            ),
            canonical_hash=canonical_hash,
            members=members,
        )
    if not isinstance(artifact, PaperChunkSet):
        raise TypeError(
            f"Unsupported paper artifact '{type(artifact).__name__}'"
        )
    members = tuple(
        _CanonicalPaperMember(
            member_id=chunk.chunk_id,
            parent_id=None,
            position=chunk.position,
            payload=(payload := _json_payload(chunk.model_dump(mode="json"))),
            content_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )
        for chunk in artifact.chunks
    )
    return _CanonicalPaperArtifact(
        artifact=artifact,
        payload=_json_payload(
            artifact.model_dump(mode="json", exclude={"chunks"})
        ),
        canonical_hash=canonical_hash,
        members=members,
    )


def _prepare_paper_source(source: PaperSourceRevision) -> tuple[str, str]:
    """Validate loaded source blobs and return its canonical payload/hash."""
    for asset in source.assets:
        blob = source.blobs.get(asset.content_hash)
        if blob is None:
            raise ValueError(
                f"Paper source is missing blob for asset '{asset.asset_id}'"
            )
        if (
            len(blob) != asset.size_bytes
            or hashlib.sha256(blob).hexdigest() != asset.content_hash
        ):
            raise ValueError(
                f"Paper source blob for asset '{asset.asset_id}' is invalid"
            )
    payload = _json_payload(source.model_dump(mode="json"))
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


_PAPER_TABLES_SQL = """
CREATE TABLE paper_sources (
    source_revision_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    source_content_hash TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    canonical_hash TEXT NOT NULL,
    asset_count INTEGER NOT NULL CHECK (asset_count > 0)
);

CREATE TABLE paper_source_assets (
    asset_id TEXT PRIMARY KEY,
    source_revision_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    page_number INTEGER,
    media_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    blob BLOB NOT NULL,
    FOREIGN KEY (source_revision_id) REFERENCES paper_sources(source_revision_id)
        ON DELETE CASCADE
);

CREATE TABLE paper_artifacts (
    artifact_id TEXT PRIMARY KEY,
    source_revision_id TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    producer_config_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    canonical_hash TEXT NOT NULL,
    member_count INTEGER NOT NULL CHECK (member_count >= 0),
    target_count INTEGER NOT NULL CHECK (target_count >= 0),
    UNIQUE (source_revision_id, artifact_kind, producer_config_hash),
    FOREIGN KEY (source_revision_id) REFERENCES paper_sources(source_revision_id)
        ON DELETE CASCADE
);

CREATE TABLE paper_artifact_members (
    artifact_id TEXT NOT NULL,
    member_id TEXT NOT NULL,
    parent_id TEXT,
    position INTEGER NOT NULL CHECK (position >= 0),
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (artifact_id, member_id),
    FOREIGN KEY (artifact_id) REFERENCES paper_artifacts(artifact_id)
        ON DELETE CASCADE
);

CREATE TABLE paper_artifact_lineage (
    artifact_id TEXT NOT NULL,
    input_artifact_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    PRIMARY KEY (artifact_id, input_artifact_id, relation),
    FOREIGN KEY (artifact_id) REFERENCES paper_artifacts(artifact_id)
        ON DELETE CASCADE,
    FOREIGN KEY (input_artifact_id) REFERENCES paper_artifacts(artifact_id)
        ON DELETE RESTRICT
);

CREATE TABLE paper_projections (
    target_id TEXT PRIMARY KEY,
    source_revision_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    member_id TEXT,
    artifact_kind TEXT NOT NULL,
    matched_text TEXT NOT NULL,
    as_of REAL NOT NULL,
    available_at REAL NOT NULL,
    source_kind TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    projection_kind TEXT NOT NULL,
    modality TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    projection_hash TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    artifact_schema_version TEXT NOT NULL,
    projection_schema_version TEXT NOT NULL,
    artifact_canonical_hash TEXT NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY (source_revision_id) REFERENCES paper_sources(source_revision_id)
        ON DELETE CASCADE,
    FOREIGN KEY (artifact_id) REFERENCES paper_artifacts(artifact_id)
        ON DELETE CASCADE,
    FOREIGN KEY (artifact_id, member_id)
        REFERENCES paper_artifact_members(artifact_id, member_id)
        ON DELETE CASCADE
);

CREATE INDEX paper_artifacts_source_kind
    ON paper_artifacts(source_revision_id, artifact_kind);
CREATE UNIQUE INDEX paper_artifact_members_sibling_position
    ON paper_artifact_members(
        artifact_id, COALESCE(parent_id, ''), position
    );
CREATE INDEX paper_projections_filters
    ON paper_projections(artifact_kind, source_kind, source_revision_id);
"""


def _migrate_schema_v3_to_v4(db: sqlite3.Connection) -> None:
    """Allow vectorless artifacts and hierarchical normalized members."""
    db.execute("PRAGMA foreign_keys = OFF")
    try:
        db.executescript(
            """
            BEGIN IMMEDIATE;

            CREATE TABLE paper_artifacts_v4 (
                artifact_id TEXT PRIMARY KEY,
                source_revision_id TEXT NOT NULL,
                artifact_kind TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                producer_config_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                canonical_hash TEXT NOT NULL,
                member_count INTEGER NOT NULL CHECK (member_count >= 0),
                target_count INTEGER NOT NULL CHECK (target_count >= 0),
                UNIQUE (source_revision_id, artifact_kind, producer_config_hash),
                FOREIGN KEY (source_revision_id)
                    REFERENCES paper_sources(source_revision_id)
                    ON DELETE CASCADE
            );

            INSERT INTO paper_artifacts_v4
            SELECT * FROM paper_artifacts;

            CREATE TABLE paper_artifact_members_v4 (
                artifact_id TEXT NOT NULL,
                member_id TEXT NOT NULL,
                parent_id TEXT,
                position INTEGER NOT NULL CHECK (position >= 0),
                payload_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                PRIMARY KEY (artifact_id, member_id),
                FOREIGN KEY (artifact_id)
                    REFERENCES paper_artifacts_v4(artifact_id)
                    ON DELETE CASCADE
            );

            INSERT INTO paper_artifact_members_v4 (
                artifact_id, member_id, parent_id, position,
                payload_json, content_hash
            )
            SELECT artifact_id, member_id, NULL, position,
                   payload_json, content_hash
            FROM paper_artifact_members;

            DROP TABLE paper_artifact_members;
            DROP TABLE paper_artifacts;
            ALTER TABLE paper_artifacts_v4 RENAME TO paper_artifacts;
            ALTER TABLE paper_artifact_members_v4
                RENAME TO paper_artifact_members;

            CREATE INDEX paper_artifacts_source_kind
                ON paper_artifacts(source_revision_id, artifact_kind);
            CREATE UNIQUE INDEX paper_artifact_members_sibling_position
                ON paper_artifact_members(
                    artifact_id, COALESCE(parent_id, ''), position
                );

            PRAGMA user_version = 4;
            COMMIT;
            """
        )
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError(
                "Stale knowledge library schema: v3 migration broke links"
            )
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.execute("PRAGMA foreign_keys = ON")


def _initialize_schema(db: sqlite3.Connection) -> None:
    """Create the current schema or reject an incompatible local database."""
    version_row = db.execute("PRAGMA user_version").fetchone()
    version = int(version_row[0])
    if version not in (0, 2, 3, _DATABASE_SCHEMA_VERSION):
        raise RuntimeError(
            "Stale knowledge library schema: database version "
            f"{version}, expected {_DATABASE_SCHEMA_VERSION}"
        )
    if version == _DATABASE_SCHEMA_VERSION:
        return
    if version == 3:
        _migrate_schema_v3_to_v4(db)
        return
    if version == 2:
        migration_sql = _PAPER_TABLES_SQL.replace(
            "CREATE TABLE ", "CREATE TABLE IF NOT EXISTS "
        ).replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ")
        db.executescript(
            f"{migration_sql}\nPRAGMA user_version = {_DATABASE_SCHEMA_VERSION};"
        )
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

        {_PAPER_TABLES_SQL}

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

    def prepare_put_paper(self, result: PaperFlowResult) -> _PreparedPaperPut:
        """Validate paper blobs and load projections eligible for reuse."""
        source = result.source_revision
        source_payload, source_canonical_hash = _prepare_paper_source(source)
        artifact_ids = (str(result.chunk_set.id), str(result.global_summary.id))
        rows = self._db.execute(
            """
            SELECT * FROM paper_projections
            WHERE artifact_id IN (?, ?)
            """,
            artifact_ids,
        ).fetchall()
        existing = {
            str(row["target_id"]): _StoredEmbedding(
                target_id=str(row["target_id"]),
                embedding_model=str(row["embedding_model"]),
                dimension=int(row["dimension"]),
                projection_hash=str(row["projection_hash"]),
                source_content_hash=str(row["source_content_hash"]),
                knowledge_schema_version=str(row["artifact_schema_version"]),
                projection_schema_version=str(row["projection_schema_version"]),
                embedding=bytes(row["embedding"]),
            )
            for row in rows
        }
        return _PreparedPaperPut(
            result=result,
            source_payload=source_payload,
            source_canonical_hash=source_canonical_hash,
            artifacts=(
                _canonical_paper_artifact(result.chunk_set),
                _canonical_paper_artifact(result.global_summary),
            ),
            existing_embeddings=existing,
        )

    def _put_paper_source(
        self,
        source: PaperSourceRevision,
        *,
        payload: str,
        canonical_hash: str,
    ) -> None:
        """Write or reuse one exact source inside the active transaction."""
        self._db.execute(
            """
            INSERT INTO paper_sources (
                source_revision_id, schema_version, source_content_hash,
                payload_json, canonical_hash, asset_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_revision_id) DO NOTHING
            """,
            (
                str(source.id),
                source.schema_version,
                source.source.content_hash,
                payload,
                canonical_hash,
                len(source.assets),
            ),
        )
        for asset in source.assets:
            self._db.execute(
                """
                INSERT INTO paper_source_assets (
                    asset_id, source_revision_id, kind, page_number,
                    media_type, content_hash, size_bytes, blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO NOTHING
                """,
                (
                    str(asset.asset_id),
                    str(source.id),
                    asset.kind,
                    asset.page_number,
                    asset.media_type,
                    asset.content_hash,
                    asset.size_bytes,
                    source.blobs[asset.content_hash],
                ),
            )

    def put_paper(
        self,
        prepared: _PreparedPaperPut,
        targets: Sequence[_RetrievalTarget],
        vectors: dict[str, tuple[bytes, int]],
        *,
        embedding_model: str,
    ) -> None:
        """Atomically persist one source, two artifacts, lineage, and vectors."""
        result = prepared.result
        source = result.source_revision
        canonical_by_id = {
            artifact.artifact.id: artifact for artifact in prepared.artifacts
        }
        targets_by_artifact: dict[UUID, list[_RetrievalTarget]] = {}
        for target in targets:
            targets_by_artifact.setdefault(target.artifact_id, []).append(
                target
            )
        if set(targets_by_artifact) != set(canonical_by_id):
            raise ValueError("paper artifacts do not have complete projections")
        try:
            self._db.execute("BEGIN IMMEDIATE")
            self._put_paper_source(
                source,
                payload=prepared.source_payload,
                canonical_hash=prepared.source_canonical_hash,
            )
            for canonical in prepared.artifacts:
                artifact = canonical.artifact
                artifact_targets = targets_by_artifact[artifact.id]
                self._db.execute(
                    """
                    INSERT INTO paper_artifacts (
                        artifact_id, source_revision_id, artifact_kind,
                        schema_version, producer_config_hash, payload_json,
                        canonical_hash, member_count, target_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(artifact_id) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        canonical_hash = excluded.canonical_hash,
                        member_count = excluded.member_count,
                        target_count = excluded.target_count
                    """,
                    (
                        str(artifact.id),
                        str(artifact.source_revision_id),
                        artifact.artifact_kind,
                        artifact.schema_version,
                        artifact.producer_config_hash,
                        canonical.payload,
                        canonical.canonical_hash,
                        len(canonical.members),
                        len(artifact_targets),
                    ),
                )
                self._db.execute(
                    "DELETE FROM paper_projections WHERE artifact_id = ?",
                    (str(artifact.id),),
                )
                self._db.execute(
                    "DELETE FROM paper_artifact_members WHERE artifact_id = ?",
                    (str(artifact.id),),
                )
                self._db.execute(
                    "DELETE FROM paper_artifact_lineage WHERE artifact_id = ?",
                    (str(artifact.id),),
                )
                for member in canonical.members:
                    self._db.execute(
                        """
                        INSERT INTO paper_artifact_members (
                            artifact_id, member_id, parent_id, position,
                            payload_json, content_hash
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(artifact.id),
                            str(member.member_id),
                            (
                                str(member.parent_id)
                                if member.parent_id is not None
                                else None
                            ),
                            member.position,
                            member.payload,
                            member.content_hash,
                        ),
                    )
            for locator in result.global_summary.derived_from:
                self._db.execute(
                    """
                    INSERT INTO paper_artifact_lineage (
                        artifact_id, input_artifact_id, relation
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        str(result.global_summary.id),
                        str(locator.artifact_id),
                        "generated_from",
                    ),
                )
            for target in targets:
                blob, dimension = vectors[target.target_id]
                canonical = canonical_by_id[target.artifact_id]
                self._db.execute(
                    """
                    INSERT INTO paper_projections (
                        target_id, source_revision_id, artifact_id, member_id,
                        artifact_kind, matched_text, as_of, available_at,
                        source_kind, citations_json, projection_kind, modality,
                        embedding_model, dimension, projection_hash,
                        source_content_hash, artifact_schema_version,
                        projection_schema_version, artifact_canonical_hash,
                        embedding
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?
                    )
                    """,
                    (
                        target.target_id,
                        str(source.id),
                        str(target.artifact_id),
                        str(target.node_id) if target.node_id else None,
                        target.artifact_kind,
                        target.text,
                        _timestamp(source.as_of, "PaperSourceRevision.as_of"),
                        _timestamp(
                            source.available_at,
                            "PaperSourceRevision.available_at",
                        ),
                        source.source.kind,
                        _json_payload(
                            [
                                citation.model_dump(mode="json")
                                for citation in target.citations
                            ]
                        ),
                        "text_embedding",
                        "text",
                        embedding_model,
                        dimension,
                        target.projection_hash,
                        source.source.content_hash,
                        canonical.artifact.schema_version,
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

    def prepare_put_paper_structure_tree(
        self,
        source: PaperSourceRevision,
        tree: PaperStructureTree,
    ) -> _PreparedPaperStructurePut:
        """Validate a vectorless structure tree against its exact source."""
        _validate_structure_tree_source(tree, source)
        source_payload, source_canonical_hash = _prepare_paper_source(source)
        return _PreparedPaperStructurePut(
            source=source,
            source_payload=source_payload,
            source_canonical_hash=source_canonical_hash,
            tree=tree,
            canonical=_canonical_paper_artifact(tree),
        )

    def put_paper_structure_tree(
        self,
        prepared: _PreparedPaperStructurePut,
    ) -> None:
        """Atomically persist one source and vectorless structure artifact."""
        tree = prepared.tree
        canonical = prepared.canonical
        try:
            self._db.execute("BEGIN IMMEDIATE")
            self._put_paper_source(
                prepared.source,
                payload=prepared.source_payload,
                canonical_hash=prepared.source_canonical_hash,
            )
            self._db.execute(
                """
                INSERT INTO paper_artifacts (
                    artifact_id, source_revision_id, artifact_kind,
                    schema_version, producer_config_hash, payload_json,
                    canonical_hash, member_count, target_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    canonical_hash = excluded.canonical_hash,
                    member_count = excluded.member_count,
                    target_count = excluded.target_count
                """,
                (
                    str(tree.id),
                    str(tree.source_revision_id),
                    tree.artifact_kind,
                    tree.schema_version,
                    tree.producer_config_hash,
                    canonical.payload,
                    canonical.canonical_hash,
                    len(canonical.members),
                ),
            )
            self._db.execute(
                "DELETE FROM paper_projections WHERE artifact_id = ?",
                (str(tree.id),),
            )
            self._db.execute(
                "DELETE FROM paper_artifact_members WHERE artifact_id = ?",
                (str(tree.id),),
            )
            self._db.execute(
                "DELETE FROM paper_artifact_lineage WHERE artifact_id = ?",
                (str(tree.id),),
            )
            for member in canonical.members:
                self._db.execute(
                    """
                    INSERT INTO paper_artifact_members (
                        artifact_id, member_id, parent_id, position,
                        payload_json, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(tree.id),
                        str(member.member_id),
                        (
                            str(member.parent_id)
                            if member.parent_id is not None
                            else None
                        ),
                        member.position,
                        member.payload,
                        member.content_hash,
                    ),
                )
            self._db.execute("COMMIT")
        except Exception:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

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

    def get_paper_source(self, source_revision_id: UUID) -> PaperSourceRevision:
        """Rehydrate one exact source revision and all referenced blobs."""
        row = self._db.execute(
            "SELECT * FROM paper_sources WHERE source_revision_id = ?",
            (str(source_revision_id),),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"Paper source revision '{source_revision_id}' not found"
            )
        payload = str(row["payload_json"])
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != str(
            row["canonical_hash"]
        ):
            raise RuntimeError(
                f"Stale paper source '{source_revision_id}': content hash mismatch"
            )
        asset_rows = self._db.execute(
            """
            SELECT * FROM paper_source_assets
            WHERE source_revision_id = ? ORDER BY asset_id
            """,
            (str(source_revision_id),),
        ).fetchall()
        if len(asset_rows) != int(row["asset_count"]):
            raise RuntimeError(
                f"Stale paper source '{source_revision_id}': expected "
                f"{row['asset_count']} assets, found {len(asset_rows)}"
            )
        try:
            parsed_payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Stale paper source '{source_revision_id}': invalid JSON"
            ) from exc
        if not isinstance(parsed_payload, dict):
            raise RuntimeError(
                f"Stale paper source '{source_revision_id}': invalid payload"
            )
        blobs: dict[str, bytes] = {}
        for asset_row in asset_rows:
            blob = bytes(asset_row["blob"])
            content_hash = str(asset_row["content_hash"])
            if (
                len(blob) != int(asset_row["size_bytes"])
                or hashlib.sha256(blob).hexdigest() != content_hash
            ):
                raise RuntimeError(
                    f"Corrupt paper asset '{asset_row['asset_id']}'"
                )
            blobs[content_hash] = blob
        parsed_payload["blobs"] = blobs
        try:
            source = PaperSourceRevision.model_validate(parsed_payload)
        except ValidationError as exc:
            raise RuntimeError(
                f"Stale paper source '{source_revision_id}': payload no longer "
                "validates"
            ) from exc
        if (
            source.id != source_revision_id
            or source.schema_version != str(row["schema_version"])
            or source.source.content_hash != str(row["source_content_hash"])
        ):
            raise RuntimeError(
                f"Stale paper source '{source_revision_id}': identity mismatch"
            )
        try:
            stored_assets = {
                UUID(str(asset_row["asset_id"])): asset_row
                for asset_row in asset_rows
            }
        except ValueError as exc:
            raise RuntimeError(
                f"Stale paper source '{source_revision_id}': invalid asset ID"
            ) from exc
        canonical_assets = {asset.asset_id: asset for asset in source.assets}
        if set(stored_assets) != set(canonical_assets):
            raise RuntimeError(
                f"Stale paper source '{source_revision_id}': asset identity "
                "mismatch"
            )
        for asset_id, asset in canonical_assets.items():
            stored = stored_assets[asset_id]
            if any(
                (
                    str(stored["source_revision_id"]) != str(source.id),
                    str(stored["kind"]) != asset.kind,
                    stored["page_number"] != asset.page_number,
                    str(stored["media_type"]) != asset.media_type,
                    str(stored["content_hash"]) != asset.content_hash,
                    int(stored["size_bytes"]) != asset.size_bytes,
                )
            ):
                raise RuntimeError(
                    f"Stale paper asset '{asset_id}': metadata mismatch"
                )
        return source

    def get_paper_artifact(self, artifact_id: UUID) -> PaperArtifact:
        """Rehydrate one validated paper artifact and normalized members."""
        row = self._db.execute(
            "SELECT * FROM paper_artifacts WHERE artifact_id = ?",
            (str(artifact_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Paper artifact '{artifact_id}' not found")
        member_rows = self._db.execute(
            """
            SELECT * FROM paper_artifact_members
            WHERE artifact_id = ?
            ORDER BY COALESCE(parent_id, ''), position, member_id
            """,
            (str(artifact_id),),
        ).fetchall()
        if len(member_rows) != int(row["member_count"]):
            raise RuntimeError(
                f"Stale paper artifact '{artifact_id}': expected "
                f"{row['member_count']} members, found {len(member_rows)}"
            )
        try:
            payload_value = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Stale paper artifact '{artifact_id}': invalid JSON"
            ) from exc
        if not isinstance(payload_value, dict):
            raise RuntimeError(
                f"Stale paper artifact '{artifact_id}': invalid payload"
            )
        artifact_kind = str(row["artifact_kind"])
        members: list[object] = []
        for member_row in member_rows:
            member_payload = str(member_row["payload_json"])
            if hashlib.sha256(
                member_payload.encode("utf-8")
            ).hexdigest() != str(member_row["content_hash"]):
                raise RuntimeError(
                    f"Stale paper artifact '{artifact_id}': member content "
                    "hash mismatch"
                )
            try:
                parsed_member = json.loads(member_payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Stale paper artifact '{artifact_id}': invalid member JSON"
                ) from exc
            if not isinstance(parsed_member, dict):
                raise RuntimeError(
                    f"Stale paper artifact '{artifact_id}': member metadata "
                    "mismatch"
                )
            if artifact_kind == "paper_structure_tree":
                metadata_matches = all(
                    (
                        parsed_member.get("node_id") == member_row["member_id"],
                        parsed_member.get("parent_id")
                        == member_row["parent_id"],
                        parsed_member.get("position") == member_row["position"],
                    )
                )
            else:
                metadata_matches = all(
                    (
                        parsed_member.get("chunk_id")
                        == member_row["member_id"],
                        member_row["parent_id"] is None,
                        parsed_member.get("position") == member_row["position"],
                    )
                )
            if not metadata_matches:
                raise RuntimeError(
                    f"Stale paper artifact '{artifact_id}': member metadata "
                    "mismatch"
                )
            members.append(parsed_member)
        if artifact_kind == "paper_chunk_set":
            payload_value["chunks"] = members
            model: (
                type[PaperChunkSet]
                | type[PaperGlobalSummary]
                | type[PaperStructureTree]
            ) = PaperChunkSet
        elif artifact_kind == "paper_summary":
            if members:
                raise RuntimeError(
                    f"Stale paper artifact '{artifact_id}': summary has members"
                )
            model = PaperGlobalSummary
        elif artifact_kind == "paper_structure_tree":
            payload_value["nodes"] = {
                str(member["node_id"]): member
                for member in members
                if isinstance(member, dict)
            }
            model = PaperStructureTree
        else:
            raise RuntimeError(
                f"Stale paper artifact '{artifact_id}': unsupported kind "
                f"'{artifact_kind}'"
            )
        full_payload = _json_payload(payload_value)
        if hashlib.sha256(full_payload.encode("utf-8")).hexdigest() != str(
            row["canonical_hash"]
        ):
            raise RuntimeError(
                f"Stale paper artifact '{artifact_id}': canonical hash mismatch"
            )
        try:
            artifact = model.model_validate(payload_value)
        except ValidationError as exc:
            raise RuntimeError(
                f"Stale paper artifact '{artifact_id}': payload no longer "
                "validates"
            ) from exc
        try:
            stored_source_revision_id = UUID(str(row["source_revision_id"]))
        except ValueError as exc:
            raise RuntimeError(
                f"Stale paper artifact '{artifact_id}': invalid source ID"
            ) from exc
        if (
            artifact.id != artifact_id
            or artifact.source_revision_id != stored_source_revision_id
            or artifact.artifact_kind != artifact_kind
            or artifact.schema_version != str(row["schema_version"])
            or artifact.producer_config_hash != str(row["producer_config_hash"])
        ):
            raise RuntimeError(
                f"Stale paper artifact '{artifact_id}': identity mismatch"
            )
        expected_target_count = (
            0
            if isinstance(artifact, PaperStructureTree)
            else 1
            if isinstance(artifact, PaperGlobalSummary)
            else len(artifact.chunks)
        )
        if int(row["target_count"]) != expected_target_count:
            raise RuntimeError(
                f"Stale paper artifact '{artifact_id}': target count mismatch"
            )
        if isinstance(artifact, PaperChunkSet):
            try:
                _validate_chunk_set_source(
                    self.get_paper_source(artifact.source_revision_id),
                    artifact,
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Stale paper artifact '{artifact_id}': source span "
                    "mismatch"
                ) from exc
        lineage_rows = self._db.execute(
            """
            SELECT input_artifact_id, relation
            FROM paper_artifact_lineage
            WHERE artifact_id = ?
            ORDER BY input_artifact_id, relation
            """,
            (str(artifact_id),),
        ).fetchall()
        expected_lineage = (
            {
                (str(locator.artifact_id), "generated_from")
                for locator in artifact.derived_from
            }
            if isinstance(artifact, PaperGlobalSummary)
            else set()
        )
        stored_lineage = {
            (str(lineage["input_artifact_id"]), str(lineage["relation"]))
            for lineage in lineage_rows
        }
        if stored_lineage != expected_lineage:
            raise RuntimeError(
                f"Stale paper artifact '{artifact_id}': lineage mismatch"
            )
        if isinstance(artifact, PaperStructureTree):
            try:
                _validate_structure_tree_source(
                    artifact,
                    self.get_paper_source(artifact.source_revision_id),
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Stale paper artifact '{artifact_id}': citation mismatch"
                ) from exc
        return artifact

    def get_paper_result(
        self,
        source_revision_id: UUID,
        *,
        chunk_set_id: UUID | None,
        summary_id: UUID | None,
    ) -> PaperFlowResult:
        """Resolve one unambiguous V1 source/chunk-set/summary combination."""
        source = self.get_paper_source(source_revision_id)

        def select(kind: str, selected_id: UUID | None) -> PaperArtifact:
            if selected_id is not None:
                artifact = self.get_paper_artifact(selected_id)
                if (
                    artifact.artifact_kind != kind
                    or artifact.source_revision_id != source_revision_id
                ):
                    raise KeyError(
                        f"Paper artifact '{selected_id}' does not belong to "
                        f"source '{source_revision_id}'"
                    )
                return artifact
            rows = self._db.execute(
                """
                SELECT artifact_id FROM paper_artifacts
                WHERE source_revision_id = ? AND artifact_kind = ?
                ORDER BY artifact_id
                """,
                (str(source_revision_id), kind),
            ).fetchall()
            if len(rows) != 1:
                raise ValueError(
                    f"Paper source '{source_revision_id}' has {len(rows)} "
                    f"'{kind}' artifacts; specify an artifact ID"
                )
            return self.get_paper_artifact(UUID(str(rows[0]["artifact_id"])))

        chunk_set = select("paper_chunk_set", chunk_set_id)
        summary = select("paper_summary", summary_id)
        if not isinstance(chunk_set, PaperChunkSet) or not isinstance(
            summary, PaperGlobalSummary
        ):
            raise RuntimeError("Stored paper artifact types are inconsistent")
        return PaperFlowResult(
            source_revision=source,
            chunk_set=chunk_set,
            global_summary=summary,
        )

    def resolve_paper_locator(
        self, locator: ArtifactLocator
    ) -> ResolvedPaperArtifact:
        """Resolve an artifact locator to its canonical aggregate or member."""
        artifact = self.get_paper_artifact(locator.artifact_id)
        if (
            artifact.source_revision_id != locator.source_revision_id
            or artifact.artifact_kind != locator.artifact_kind
        ):
            raise KeyError(
                "Paper artifact locator metadata does not match storage"
            )
        if locator.member_id is None:
            return artifact
        if isinstance(artifact, PaperStructureTree):
            try:
                node = artifact.nodes[locator.member_id]
            except KeyError as exc:
                raise KeyError(
                    f"Paper structure node '{locator.member_id}' not found"
                ) from exc
            source = self.get_paper_source(artifact.source_revision_id)
            pages = {
                page.page_number: page.text for page in source.parsed.pages
            }
            content: list[str] = []
            seen: set[int] = set()
            for citation in node.citations:
                page_number = citation.page
                if page_number is None or page_number in seen:
                    continue
                page_text = pages.get(page_number)
                if page_text is None:
                    raise RuntimeError(
                        "Stored paper structure-tree citation is unresolved"
                    )
                seen.add(page_number)
                content.append(page_text)
            return node.model_copy(update={"content": "\n\n".join(content)})
        if not isinstance(artifact, PaperChunkSet):
            raise KeyError("Paper artifact does not have resolvable members")
        for chunk in artifact.chunks:
            if chunk.chunk_id == locator.member_id:
                return chunk
        raise KeyError(f"Paper chunk '{locator.member_id}' not found")

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
                    owner_kind="knowledge",
                    item_id=UUID(str(row["item_id"])),
                    node_id=UUID(str(node_value)) if node_value else None,
                    item_type=str(row["item_type"]),
                    source_revision_id=None,
                    artifact_kind=str(row["item_type"]),
                    projection_kind="text_embedding",
                    projection_version=str(row["projection_schema_version"]),
                    embedding_model=str(row["embedding_model"]),
                    projection_hash=str(row["projection_hash"]),
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

        orphan_paper = self._db.execute(
            """
            SELECT p.target_id FROM paper_projections AS p
            LEFT JOIN paper_sources AS s
                ON s.source_revision_id = p.source_revision_id
            LEFT JOIN paper_artifacts AS a ON a.artifact_id = p.artifact_id
            WHERE s.source_revision_id IS NULL OR a.artifact_id IS NULL
            LIMIT 1
            """
        ).fetchone()
        if orphan_paper is not None:
            raise RuntimeError(
                f"Stale paper projection '{orphan_paper['target_id']}': "
                "source or artifact is missing"
            )
        orphan_member = self._db.execute(
            """
            SELECT p.target_id FROM paper_projections AS p
            LEFT JOIN paper_artifact_members AS m
                ON m.artifact_id = p.artifact_id
               AND m.member_id = p.member_id
            WHERE p.member_id IS NOT NULL AND m.member_id IS NULL
            LIMIT 1
            """
        ).fetchone()
        if orphan_member is not None:
            raise RuntimeError(
                f"Stale paper projection '{orphan_member['target_id']}': "
                "artifact member is missing"
            )
        incomplete_artifact = self._db.execute(
            """
            SELECT a.artifact_id, a.member_count,
                   COUNT(DISTINCT m.member_id) AS actual_members,
                   a.target_count,
                   COUNT(DISTINCT p.target_id) AS actual_targets
            FROM paper_artifacts AS a
            LEFT JOIN paper_artifact_members AS m
                ON m.artifact_id = a.artifact_id
            LEFT JOIN paper_projections AS p
                ON p.artifact_id = a.artifact_id
            GROUP BY a.artifact_id, a.member_count, a.target_count
            HAVING actual_members != a.member_count
                OR actual_targets != a.target_count
            LIMIT 1
            """
        ).fetchone()
        if incomplete_artifact is not None:
            raise RuntimeError(
                f"Stale paper artifact '{incomplete_artifact['artifact_id']}': "
                f"expected {incomplete_artifact['member_count']} members and "
                f"{incomplete_artifact['target_count']} projections"
            )
        paper_rows = self._db.execute(
            """
            SELECT p.*, a.canonical_hash AS current_canonical_hash,
                   a.schema_version AS current_schema_version,
                   s.source_content_hash AS current_source_content_hash
            FROM paper_projections AS p
            JOIN paper_artifacts AS a ON a.artifact_id = p.artifact_id
            JOIN paper_sources AS s
                ON s.source_revision_id = p.source_revision_id
            ORDER BY p.target_id
            """
        ).fetchall()
        paper_artifacts: dict[UUID, PaperArtifact] = {}
        paper_sources: dict[UUID, PaperSourceRevision] = {}
        for row in paper_rows:
            target_id = str(row["target_id"])
            dimension = int(row["dimension"])
            if str(row["embedding_model"]) != embedding_model:
                raise RuntimeError(
                    f"Stale paper projection '{target_id}': embedding model "
                    "changed; re-put the paper result"
                )
            if (
                embedding_dimensions is not None
                and dimension != embedding_dimensions
            ):
                raise RuntimeError(
                    f"Stale paper projection '{target_id}': embedding dimension "
                    "changed; re-put the paper result"
                )
            if (
                str(row["projection_kind"]) != "text_embedding"
                or str(row["modality"]) != "text"
                or str(row["projection_schema_version"])
                != _PROJECTION_SCHEMA_VERSION
                or str(row["artifact_schema_version"])
                != str(row["current_schema_version"])
                or str(row["artifact_canonical_hash"])
                != str(row["current_canonical_hash"])
                or str(row["source_content_hash"])
                != str(row["current_source_content_hash"])
            ):
                raise RuntimeError(
                    f"Stale paper projection '{target_id}': projection, source, "
                    "or artifact metadata changed; re-put the paper result"
                )
            try:
                artifact_id = UUID(str(row["artifact_id"]))
                source_id = UUID(str(row["source_revision_id"]))
                member_id = (
                    UUID(str(row["member_id"]))
                    if row["member_id"] is not None
                    else None
                )
                citations_value = json.loads(str(row["citations_json"]))
                if not isinstance(citations_value, list):
                    raise ValueError
                stored_citations = [
                    Citation.model_validate(citation_value)
                    for citation_value in citations_value
                ]
            except (TypeError, ValueError, ValidationError) as exc:
                raise RuntimeError(
                    f"Corrupt paper projection '{target_id}': invalid metadata"
                ) from exc

            artifact = paper_artifacts.get(artifact_id)
            if artifact is None:
                artifact = self.get_paper_artifact(artifact_id)
                paper_artifacts[artifact_id] = artifact
            source = paper_sources.get(source_id)
            if source is None:
                source = self.get_paper_source(source_id)
                paper_sources[source_id] = source
            artifact_kind = str(row["artifact_kind"])
            if (
                artifact.source_revision_id != source_id
                or artifact.artifact_kind != artifact_kind
                or source.source.kind != str(row["source_kind"])
                or _timestamp(source.as_of, "PaperSourceRevision.as_of")
                != float(row["as_of"])
                or _timestamp(
                    source.available_at,
                    "PaperSourceRevision.available_at",
                )
                != float(row["available_at"])
            ):
                raise RuntimeError(
                    f"Stale paper projection '{target_id}': canonical locator "
                    "or source evidence mismatch"
                )

            if member_id is None:
                if not isinstance(artifact, PaperGlobalSummary):
                    raise RuntimeError(
                        f"Stale paper projection '{target_id}': chunk-set "
                        "aggregate is not searchable"
                    )
                expected_target_id = f"artifact:{artifact.id}"
                expected_text = artifact.summary
                expected_citations = [
                    Citation(
                        source_id=str(source_id),
                        page=citation.page_number,
                        quote=citation.quote,
                    )
                    for citation in artifact.citations
                ]
            else:
                if not isinstance(artifact, PaperChunkSet):
                    raise RuntimeError(
                        f"Stale paper projection '{target_id}': summary has a "
                        "search member"
                    )
                chunk = next(
                    (
                        value
                        for value in artifact.chunks
                        if value.chunk_id == member_id
                    ),
                    None,
                )
                if chunk is None:
                    raise RuntimeError(
                        f"Stale paper projection '{target_id}': canonical "
                        "chunk is missing"
                    )
                expected_target_id = (
                    f"artifact-member:{artifact.id}:{chunk.chunk_id}"
                )
                expected_text = chunk.text
                expected_citations = [
                    Citation(
                        source_id=str(source_id),
                        page=span.page_number,
                        char_offset=span.start_char,
                        quote=chunk.text[:500],
                    )
                    for span in chunk.source_spans
                ]
            expected_projection_hash = hashlib.sha256(
                expected_text.encode("utf-8")
            ).hexdigest()
            if any(
                (
                    target_id != expected_target_id,
                    str(row["matched_text"]) != expected_text,
                    str(row["projection_hash"]) != expected_projection_hash,
                    stored_citations != expected_citations,
                )
            ):
                raise RuntimeError(
                    f"Stale paper projection '{target_id}': canonical text, "
                    "hash, or citations mismatch"
                )
            record = _IndexRecord(
                target_id=target_id,
                owner_kind="paper",
                item_id=artifact_id,
                node_id=member_id,
                item_type=artifact_kind,
                source_revision_id=source_id,
                artifact_kind=artifact_kind,
                projection_kind=str(row["projection_kind"]),
                projection_version=str(row["projection_schema_version"]),
                embedding_model=str(row["embedding_model"]),
                projection_hash=expected_projection_hash,
                matched_text=expected_text,
                as_of=float(row["as_of"]),
                available_at=float(row["available_at"]),
                source_kind=str(row["source_kind"]),
                confidence="high",
                tags=frozenset(),
                tree_id=None,
                dimension=dimension,
                embedding=bytes(row["embedding"]),
            )
            records.append(record)
        return records

    def close(self) -> None:
        """Close the owned SQLite connection."""
        self._db.close()
