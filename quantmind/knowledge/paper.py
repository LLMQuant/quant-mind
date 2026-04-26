"""Paper knowledge schema (output of paper_flow)."""

from typing import Literal

from pydantic import Field

from quantmind.knowledge._base import KnowledgeItem


class Paper(KnowledgeItem):
    """A research paper extraction (one paper -> one Paper instance)."""

    item_type: Literal["paper"] = "paper"

    summary: str
    methodology: str | None = None
    key_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    asset_classes: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    arxiv_id: str | None = None
