"""Base types for the quantmind.knowledge package.

`KnowledgeItem` is the frozen Pydantic v2 base every domain schema (Paper,
News, Earnings, ...) inherits. Subclasses are returned as Agents SDK
`output_type=`, so they must be both serialisable and strict.

The `as_of` field is mandatory: financial knowledge is only useful when its
time-of-validity is explicit.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A pointer back to the source span an extracted fact came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    page: int | None = None
    char_offset: int | None = None
    quote: str | None = Field(default=None, max_length=500)


class KnowledgeItem(BaseModel):
    """Base schema for every quantmind knowledge type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citations: list[Citation] = Field(default_factory=list)
    as_of: datetime = Field(..., description="Information cutoff time.")
    confidence: Literal["low", "medium", "high"] = "medium"

    item_type: str
    source: str | None = None
    extraction_method: str | None = None

    tags: list[str] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)
