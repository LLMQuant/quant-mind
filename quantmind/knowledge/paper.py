"""Paper knowledge schemas.

A whole paper is `TreeKnowledge` (sections + subsections); the distilled
summary card is `PaperKnowledgeCard` (`FlattenKnowledge`). They are separate
items linked by ``PaperKnowledgeCard.paper_id == Paper.id``.

`paper_flow` (PR5) typically produces a `Paper` first (sectioning + per-node
summarisation), then a `PaperKnowledgeCard` derived from the root summary.
"""

from typing import Literal
from uuid import UUID

from pydantic import Field

from quantmind.knowledge._flatten import FlattenKnowledge
from quantmind.knowledge._tree import TreeKnowledge


class Paper(TreeKnowledge):
    """A research paper as a tree of sections.

    The tree's nodes carry per-section ``summary`` (for navigation) and, on
    leaves, the section ``content`` (Markdown). Top-level metadata
    (``arxiv_id``, ``authors``) lives here on the tree itself.
    """

    item_type: Literal["paper"] = "paper"
    arxiv_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    asset_classes: list[str] = Field(default_factory=list)


class PaperKnowledgeCard(FlattenKnowledge):
    """Distilled summary card of a `Paper`.

    The store layer keys this off ``paper_id`` so the card and its tree can
    be retrieved together. The card is the right shape for tagging,
    filtering, and dashboard surfaces; deep questions go to the tree.
    """

    item_type: Literal["paper_card"] = "paper_card"

    paper_id: UUID
    summary: str
    methodology: str | None = None
    key_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    asset_classes: list[str] = Field(default_factory=list)

    def embedding_text(self) -> str:
        return f"{self.summary}\n{' '.join(self.key_findings)}"
