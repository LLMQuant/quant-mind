"""Base types for the quantmind.knowledge data standard.

Three shapes share `BaseKnowledge`:

- `FlattenKnowledge` — atomic cards (`News`, `Earnings`, `PaperKnowledgeCard`, ...)
- `TreeKnowledge` — hierarchical artifacts (`Paper`, `EarningsCallTranscript`, ...)
- `GraphKnowledge` — cross-item edges (placeholder, future PR)

The `as_of` field is mandatory by design: financial knowledge is only useful
when its time-of-validity is explicit. `source` and `extraction` are typed
references (not bare strings) so dedup, audit, and re-runs all have stable
keys.

Subclasses MUST override `embedding_text()` to declare what string the store
layer should embed for them. The contract is enforced at runtime, not at
type-check time, because `BaseKnowledge` itself can be referenced as a
generic return type without forcing every consumer to know about embeddings.
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A pointer back to the source span an extracted fact came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    page: int | None = None
    char_offset: int | None = None
    quote: str | None = Field(default=None, max_length=500)
    # When the citation points into a TreeKnowledge, anchor to a specific node.
    tree_id: UUID | None = None
    node_id: UUID | None = None


class SourceRef(BaseModel):
    """Typed provenance reference. Replaces a bare ``source: str``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "arxiv", "http", "doi", "local", "rss", "transcript", "manual"
    ]
    uri: str | None = None
    fetched_at: datetime | None = None
    content_hash: str | None = None  # sha256 of fetched bytes; dedup key.


class ExtractionRef(BaseModel):
    """Records which flow + model produced this knowledge item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    flow: str
    model: str
    run_id: UUID | None = None
    extracted_at: datetime


class BaseKnowledge(BaseModel):
    """Root of every quantmind knowledge type.

    All three shapes (Flatten / Tree / Graph) share this field set; subclasses
    add domain-specific payload. Subclasses MUST override `embedding_text()`
    to declare what string the store layer should embed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Identity
    id: UUID = Field(default_factory=uuid4)
    item_type: str
    schema_version: str = "1.0"

    # Time (financial mandate)
    as_of: datetime = Field(..., description="Information cutoff time.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Provenance (no bare strings)
    source: SourceRef
    extraction: ExtractionRef | None = None

    # Trust
    confidence: Literal["low", "medium", "high"] = "medium"

    # Citations & tags
    citations: list[Citation] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)

    def embedding_text(self) -> str:
        """Return the canonical string the store should embed for this item.

        Flatten subclasses: typically ``summary + key attrs``.
        Tree subclasses: typically ``root.title + root.summary``.
        Subclasses MUST override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.embedding_text() must be overridden"
        )
