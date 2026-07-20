"""Public domain types for semantic knowledge retrieval."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantmind.knowledge import (
    ArtifactLocator,
    Citation,
    PaperArtifactKind,
    SourceRef,
)


class SemanticQuery(BaseModel):
    """A financial-time-aware semantic query over canonical knowledge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    artifact_kinds: list[PaperArtifactKind] | None = None
    item_types: list[str] | None = None
    source_kinds: (
        list[
            Literal[
                "arxiv",
                "http",
                "doi",
                "local",
                "rss",
                "transcript",
                "manual",
            ]
        ]
        | None
    ) = None
    confidence: Literal["low", "medium", "high"] | None = None
    tags: list[str] | None = None
    tree_id: UUID | None = None
    as_of_before: datetime | None = None
    available_at_before: datetime | None = None
    top_k: int = Field(default=10, ge=1)

    @field_validator("text")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query text must not be blank")
        return stripped

    @field_validator("as_of_before", "available_at_before")
    @classmethod
    def _cutoffs_must_be_timezone_aware(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("financial-time cutoffs must be timezone-aware")
        return value


class SearchProjection(BaseModel):
    """Rebuildable projection details used to rank one semantic hit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["text_embedding"] = "text_embedding"
    version: str
    modality: Literal["text"] = "text"
    model: str
    dimensions: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class SemanticHit(BaseModel):
    """Auditable evidence returned by semantic ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    locator: ArtifactLocator
    projection: SearchProjection
    item_id: UUID
    node_id: UUID | None
    item_type: str
    score: float
    matched_text: str
    as_of: datetime
    available_at: datetime | None
    source: SourceRef
    citations: list[Citation]
