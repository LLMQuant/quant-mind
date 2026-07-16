"""Canonical serialization and searchable projection rules."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from pydantic import ValidationError

from quantmind.knowledge import (
    BaseKnowledge,
    Earnings,
    Factor,
    FlattenKnowledge,
    News,
    Paper,
    PaperKnowledgeCard,
    Thesis,
    TreeKnowledge,
)

_PROJECTION_SCHEMA_VERSION = "1"

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
class _Projection:
    """One stable item or node target to embed."""

    target_id: str
    node_id: UUID | None
    text: str
    projection_hash: str
    tree_id: UUID | None


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


def _project_knowledge(item: BaseKnowledge) -> list[_Projection]:
    """Project flat items and trees at the issue-defined retrieval grain."""
    if not isinstance(item, (FlattenKnowledge, TreeKnowledge)):
        raise TypeError(
            "LocalKnowledgeLibrary supports FlattenKnowledge and TreeKnowledge"
        )

    root_text = item.embedding_text()
    if not root_text:
        raise ValueError("knowledge embedding_text() must not be empty")
    tree_id = item.id if isinstance(item, TreeKnowledge) else None
    projections = [
        _Projection(
            target_id=f"item:{item.id}",
            node_id=None,
            text=root_text,
            projection_hash=hashlib.sha256(
                root_text.encode("utf-8")
            ).hexdigest(),
            tree_id=tree_id,
        )
    ]

    if isinstance(item, TreeKnowledge):
        if item.root_node_id not in item.nodes:
            raise ValueError("tree root_node_id is not present in nodes")
        for node_id, node in sorted(
            item.nodes.items(), key=lambda pair: str(pair[0])
        ):
            if node_id != node.node_id:
                raise ValueError("tree node map key does not match node_id")
            if node_id == item.root_node_id:
                continue
            text = node.embedding_text()
            if not text:
                raise ValueError("tree node embedding_text() must not be empty")
            projections.append(
                _Projection(
                    target_id=f"node:{item.id}:{node_id}",
                    node_id=node_id,
                    text=text,
                    projection_hash=hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    tree_id=item.id,
                )
            )
    return projections
