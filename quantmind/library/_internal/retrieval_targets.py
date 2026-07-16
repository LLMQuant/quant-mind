"""Map canonical knowledge to stable searchable text targets."""

import hashlib
from dataclasses import dataclass
from uuid import UUID

from quantmind.knowledge import BaseKnowledge, FlattenKnowledge, TreeKnowledge

_PROJECTION_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class _RetrievalTarget:
    """One stable item or non-root node target to embed."""

    target_id: str
    node_id: UUID | None
    text: str
    projection_hash: str
    tree_id: UUID | None


def _project_knowledge(item: BaseKnowledge) -> list[_RetrievalTarget]:
    """Project flat items and trees at the issue-defined retrieval grain."""
    if not isinstance(item, (FlattenKnowledge, TreeKnowledge)):
        raise TypeError(
            "LocalKnowledgeLibrary supports FlattenKnowledge and TreeKnowledge"
        )

    root_text = item.embedding_text()
    if not root_text:
        raise ValueError("knowledge embedding_text() must not be empty")
    tree_id = item.id if isinstance(item, TreeKnowledge) else None
    targets = [
        _RetrievalTarget(
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
            targets.append(
                _RetrievalTarget(
                    target_id=f"node:{item.id}:{node_id}",
                    node_id=node_id,
                    text=text,
                    projection_hash=hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    tree_id=item.id,
                )
            )
    return targets
