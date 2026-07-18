"""LLM-facing draft schema for paper_flow + assembly into ``Paper``.

The Agent produces a ``DraftPaper``: a nested tree carrying only content
(no ids). ``assemble_paper`` then mints every UUID and builds the real
``Paper``. Keeping ids out of the LLM's hands makes id typos, id reuse, and
dangling references impossible by construction.
"""

from datetime import date, datetime
from typing import TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from quantmind.knowledge import (
    Citation,
    ExtractionRef,
    Paper,
    SourceRef,
    TreeNode,
)


class DraftCitation(BaseModel):
    """A citation the LLM attaches to a node — no id references."""

    model_config = ConfigDict(extra="forbid")

    quote: str | None = Field(default=None, max_length=500)
    page: int | None = None
    char_offset: int | None = None


class DraftNode(BaseModel):
    """A draft section node. Nests children directly; carries no ids."""

    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str
    content: str | None = None
    citations: list[DraftCitation] = Field(default_factory=list)
    children: list["DraftNode"] = Field(default_factory=list)


class DraftPaper(BaseModel):
    """Top-level LLM output — the paper as the root of a nested section tree.

    ``DraftPaper`` *is* the root node: it carries the paper's ``title`` /
    ``summary`` (plus optional ``content`` / ``citations``) and its
    ``children`` are the paper's top-level sections, each of which may nest
    further. This matches how the model naturally emits a document, rather
    than forcing a single ``root`` field.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str
    published_date: date | None = None
    arxiv_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    asset_classes: list[str] = Field(default_factory=list)
    content: str | None = None
    citations: list[DraftCitation] = Field(default_factory=list)
    children: list["DraftNode"] = Field(default_factory=list)


P = TypeVar("P", bound=Paper)


def assemble_paper(
    draft: DraftPaper,
    *,
    source: SourceRef,
    source_id: str,
    as_of: datetime,
    extraction: ExtractionRef,
    out_type: type[P],
) -> P:
    """Walk ``draft``, mint UUIDs, and build a fully-wired ``Paper``.

    Every node gets a fresh ``uuid4()``; ``parent_id`` / ``children_ids`` /
    ``root_node_id`` and each citation's ``node_id`` / ``tree_id`` are set by
    this function so the LLM never has to emit an id.
    """
    paper_id = uuid4()
    nodes: dict[UUID, TreeNode] = {}

    def _build(node: DraftNode, parent_id: UUID | None, position: int) -> UUID:
        node_id = uuid4()
        children_ids = [
            _build(child, node_id, i) for i, child in enumerate(node.children)
        ]
        citations = [
            Citation(
                source_id=source_id,
                page=c.page,
                char_offset=c.char_offset,
                quote=c.quote,
                tree_id=paper_id,
                node_id=node_id,
            )
            for c in node.citations
        ]
        nodes[node_id] = TreeNode(
            node_id=node_id,
            parent_id=parent_id,
            position=position,
            title=node.title,
            summary=node.summary,
            content=node.content,
            citations=citations,
            children_ids=children_ids,
        )
        return node_id

    root_draft = DraftNode(
        title=draft.title,
        summary=draft.summary,
        content=draft.content,
        citations=draft.citations,
        children=draft.children,
    )
    root_id = _build(root_draft, None, 0)

    return out_type(
        id=paper_id,
        as_of=as_of,
        source=source,
        extraction=extraction,
        root_node_id=root_id,
        nodes=nodes,
        arxiv_id=draft.arxiv_id,
        authors=list(draft.authors),
        asset_classes=list(draft.asset_classes),
    )
