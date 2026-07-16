"""Canonical serialization and searchable projection rules."""

import hashlib
import json
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


def _canonical_payload(item: BaseKnowledge) -> tuple[str, str, str]:
    """Serialize a supported canonical item and return its stable hash."""
    knowledge_class = f"{type(item).__module__}:{type(item).__qualname__}"
    if knowledge_class not in _KNOWLEDGE_CLASSES:
        raise TypeError(
            f"Unsupported knowledge type '{type(item).__name__}'; "
            "only canonical quantmind.knowledge types can be persisted"
        )
    payload = json.dumps(
        item.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return knowledge_class, payload, content_hash


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
