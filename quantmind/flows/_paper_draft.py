"""LLM-facing draft schema for ``paper_flow`` + lift into canonical ``Paper``.

The canonical :class:`quantmind.knowledge.Paper` is the *store* schema: a flat
``dict[UUID, TreeNode]`` keyed by UUID for O(1) lookup and stable dedup keys.
That shape is hostile to LLM structured output for two reasons:

1. A ``dict`` field serialises to ``additionalProperties``, which the Agents
   SDK's strict-JSON-schema mode rejects outright.
2. Even with strict mode off, models do not reliably emit RFC-4122 UUIDs, so
   ``UUID`` id fields fail Pydantic validation on free-form ids like
   ``"intro_node"``.

So the agent targets :class:`PaperDraft` instead — a nested tree of
:class:`PaperDraftNode` with plain-string-free structure (children are
embedded, not referenced by id) that is strict-schema clean. ``draft_to_paper``
then assigns real UUIDs, wires ``parent_id`` / ``children_ids``, and injects
provenance (``source``, ``arxiv_id``, ``authors``) the flow already knows from
the fetch layer rather than trusting the model to hallucinate it.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from quantmind.knowledge import (
    Citation,
    ExtractionRef,
    Paper,
    SourceRef,
    TreeNode,
)

# Map the ``source`` discriminator emitted by ``_fetch_and_format`` onto the
# ``SourceRef.kind`` literal. ``inline`` (RawText) has no external origin, so it
# records as a manual entry.
_SOURCE_KIND: dict[str, str] = {
    "arxiv": "arxiv",
    "web": "http",
    "local": "local",
    "inline": "manual",
}

# ``Citation.quote`` caps at 500 chars; models routinely over-quote.
_QUOTE_MAX = 500


class DraftCitation(BaseModel):
    """A citation as emitted by the model (no tree/node anchors yet)."""

    model_config = ConfigDict(extra="ignore")

    source_id: str
    page: int | None = None
    char_offset: int | None = None
    quote: str | None = None


class PaperDraftNode(BaseModel):
    """One section as emitted by the model.

    Children are embedded directly (a true tree), so the model never has to
    keep a flat id map consistent — eliminating dangling-reference and
    duplicate-id failure modes. ``draft_to_paper`` assigns identity.
    """

    model_config = ConfigDict(extra="ignore")

    title: str
    summary: str
    content: str | None = None
    citations: list[DraftCitation] = Field(default_factory=list)
    children: list["PaperDraftNode"] = Field(default_factory=list)


class PaperDraft(BaseModel):
    """Strict-schema-safe extraction target for ``paper_flow``.

    Carries only what the model can author from the text. Provenance and
    identity are supplied by :func:`draft_to_paper`.
    """

    model_config = ConfigDict(extra="ignore")

    root: PaperDraftNode
    arxiv_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    asset_classes: list[str] = Field(default_factory=list)


def _to_citations(drafts: list[DraftCitation]) -> list[Citation]:
    return [
        Citation(
            source_id=d.source_id,
            page=d.page,
            char_offset=d.char_offset,
            quote=None if d.quote is None else d.quote[:_QUOTE_MAX],
        )
        for d in drafts
    ]


def _source_ref(source_meta: dict[str, Any]) -> SourceRef:
    origin = source_meta.get("source", "manual")
    kind = _SOURCE_KIND.get(origin, "manual")
    if origin == "arxiv":
        aid = source_meta.get("arxiv_id")
        uri = f"arxiv:{aid}" if aid else None
    elif origin == "web":
        uri = source_meta.get("url")
    elif origin == "local":
        uri = source_meta.get("path")
    else:
        uri = None
    return SourceRef(kind=kind, uri=uri)  # type: ignore[arg-type]


def draft_to_paper(
    draft: PaperDraft,
    *,
    source_meta: dict[str, Any],
    model: str,
) -> Paper:
    """Lift a validated ``PaperDraft`` into a canonical ``Paper``.

    Args:
        draft: The model's nested extraction output.
        source_meta: The ``(_, meta)`` dict from ``_fetch_and_format`` — the
            authoritative provenance (origin, arxiv id, authors, published
            date) known to the flow.
        model: Model identifier recorded on the ``ExtractionRef``.

    Returns:
        A fully-formed ``Paper`` with UUID identity and injected provenance.
    """
    nodes: dict[Any, TreeNode] = {}

    def build(dn: PaperDraftNode, parent_id: Any, position: int) -> Any:
        node_id = uuid4()
        children_ids = [
            build(child, node_id, pos) for pos, child in enumerate(dn.children)
        ]
        nodes[node_id] = TreeNode(
            node_id=node_id,
            parent_id=parent_id,
            position=position,
            title=dn.title,
            summary=dn.summary,
            content=dn.content,
            citations=_to_citations(dn.citations),
            children_ids=children_ids,
        )
        return node_id

    root_id = build(draft.root, None, 0)

    published = source_meta.get("published_at")
    as_of = (
        published
        if isinstance(published, datetime)
        else datetime.now(timezone.utc)
    )
    authors = source_meta.get("authors") or draft.authors

    return Paper(
        as_of=as_of,
        source=_source_ref(source_meta),
        extraction=ExtractionRef(
            flow="paper_flow",
            model=model,
            extracted_at=datetime.now(timezone.utc),
        ),
        root_node_id=root_id,
        nodes=nodes,
        arxiv_id=source_meta.get("arxiv_id") or draft.arxiv_id,
        authors=list(authors),
        asset_classes=draft.asset_classes,
    )
