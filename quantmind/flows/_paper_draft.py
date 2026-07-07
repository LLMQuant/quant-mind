"""LLM-facing draft schema for paper_flow + assembly into ``Paper``.

The Agent produces a ``DraftPaper``: a nested tree carrying only content
(no ids). ``assemble_paper`` then mints every UUID and builds the real
``Paper``. Keeping ids out of the LLM's hands makes id typos, id reuse, and
dangling references impossible by construction.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


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
    """Top-level LLM output: paper metadata + the root draft node."""

    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str
    published_date: date | None = None
    arxiv_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    asset_classes: list[str] = Field(default_factory=list)
    root: DraftNode
