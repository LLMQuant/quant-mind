"""Map canonical knowledge to stable searchable text targets."""

import hashlib
from dataclasses import dataclass
from uuid import UUID

from quantmind.knowledge import (
    BaseKnowledge,
    Citation,
    Earnings,
    Factor,
    FlattenKnowledge,
    News,
    PaperFlowResult,
    Thesis,
    TreeKnowledge,
)

_PROJECTION_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class _RetrievalTarget:
    """One stable item or non-root node target to embed."""

    target_id: str
    artifact_id: UUID
    artifact_kind: str
    source_revision_id: UUID | None
    node_id: UUID | None
    text: str
    projection_hash: str
    tree_id: UUID | None
    citations: tuple[Citation, ...] = ()


def _project_knowledge(item: BaseKnowledge) -> list[_RetrievalTarget]:
    """Project flat items and trees at the issue-defined retrieval grain."""
    if not isinstance(item, (FlattenKnowledge, TreeKnowledge)):
        raise TypeError(
            "LocalKnowledgeLibrary supports FlattenKnowledge and TreeKnowledge"
        )

    root_text = _knowledge_text(item)
    if not root_text:
        raise ValueError("knowledge text projection must not be empty")
    tree_id = item.id if isinstance(item, TreeKnowledge) else None
    targets = [
        _RetrievalTarget(
            target_id=f"item:{item.id}",
            artifact_id=item.id,
            artifact_kind=item.item_type,
            source_revision_id=None,
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
            text = f"{node.title}\n{node.summary}"
            if not text:
                raise ValueError("tree node text projection must not be empty")
            targets.append(
                _RetrievalTarget(
                    target_id=f"node:{item.id}:{node_id}",
                    artifact_id=item.id,
                    artifact_kind=item.item_type,
                    source_revision_id=None,
                    node_id=node_id,
                    text=text,
                    projection_hash=hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    tree_id=item.id,
                )
            )
    return targets


def _project_paper(result: PaperFlowResult) -> list[_RetrievalTarget]:
    """Project a cited summary and every non-empty chunk for text search."""
    source_id = result.source_revision.id
    summary = result.global_summary
    targets = [
        _RetrievalTarget(
            target_id=f"artifact:{summary.id}",
            artifact_id=summary.id,
            artifact_kind=summary.artifact_kind,
            source_revision_id=source_id,
            node_id=None,
            text=summary.summary,
            projection_hash=hashlib.sha256(
                summary.summary.encode("utf-8")
            ).hexdigest(),
            tree_id=None,
            citations=tuple(
                Citation(
                    source_id=str(source_id),
                    page=citation.page_number,
                    quote=citation.quote,
                )
                for citation in summary.citations
            ),
        )
    ]
    for chunk in result.chunk_set.chunks:
        if not chunk.text.strip():
            continue
        targets.append(
            _RetrievalTarget(
                target_id=f"artifact-member:{result.chunk_set.id}:{chunk.chunk_id}",
                artifact_id=result.chunk_set.id,
                artifact_kind=result.chunk_set.artifact_kind,
                source_revision_id=source_id,
                node_id=chunk.chunk_id,
                text=chunk.text,
                projection_hash=hashlib.sha256(
                    chunk.text.encode("utf-8")
                ).hexdigest(),
                tree_id=None,
                citations=tuple(
                    Citation(
                        source_id=str(source_id),
                        page=span.page_number,
                        char_offset=span.start_char,
                        quote=chunk.text[:500],
                    )
                    for span in chunk.source_spans
                ),
            )
        )
    if len(targets) != len(result.chunk_set.chunks) + 1:
        raise ValueError(
            "every paper chunk must have a required text projection"
        )
    return targets


def _knowledge_text(item: BaseKnowledge) -> str:
    """Select searchable text at the indexing boundary, not in the model."""
    if isinstance(item, News):
        entities = ", ".join(item.entities)
        return f"{item.headline}\n{item.event_type}\n{entities}".strip()
    if isinstance(item, Earnings):
        guidance = item.guidance or ""
        return f"{item.ticker} {item.period} earnings\n{guidance}".strip()
    if isinstance(item, Factor):
        scope = item.universe or "unspecified"
        return f"factor {item.factor_name} on {scope}"
    if isinstance(item, Thesis):
        return item.claim
    if isinstance(item, TreeKnowledge):
        root = item.root()
        return f"{root.title}\n{root.summary}"
    raise TypeError(
        f"Unsupported knowledge projection type '{type(item).__name__}'"
    )
