"""Base types for the quantmind.knowledge data standard.

Three conventional shapes share `BaseKnowledge`:

- `FlattenKnowledge` — atomic cards (`News`, `Earnings`, `Factor`, ...)
- `TreeKnowledge` — hierarchical conventional artifacts
- `GraphKnowledge` — cross-item edges (placeholder, future PR)

Source-first paper revisions and artifacts use separate immutable models rather
than inheriting `BaseKnowledge`.

The `as_of` field is mandatory by design: financial knowledge is only useful
when its information cutoff is explicit. Optional `available_at` records when
the source became observable so research can prevent look-ahead independently
from information time. `source` and `extraction` are typed references (not bare
strings) so dedup, audit, and re-runs all have stable keys.

Canonical knowledge does not choose text or store vectors for a retrieval
method. Search projections are rebuildable library-owned derived data.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self


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


class ArtifactMeta(BaseModel):
    """Minimal provenance for a derived, self-contained artifact.

    A derived artifact (for example a page-preserving structure tree) is not
    canonical ``BaseKnowledge``: it is rebuildable from its source. It still
    needs the light provenance carried here to be stored and time-queried on its
    own — an information cutoff (``as_of``), a typed light source reference, and
    the exact content hash of the source revision it was derived from. Kept
    deliberately small so several artifact shapes can mix it in without
    inheriting the full canonical-knowledge field set.

    This provenance is metadata, never identity: it must stay out of an
    artifact's ``id`` and ``content_hash`` so a rebuild at a different wall-clock
    time yields the identical artifact.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime = Field(..., description="Information cutoff time.")
    source: SourceRef
    source_title: str | None = None
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class BaseKnowledge(BaseModel):
    """Root of every quantmind knowledge type.

    All three shapes (Flatten / Tree / Graph) share this field set; subclasses
    add domain-specific payload. Retrieval-specific text selection belongs to
    the library indexing boundary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Identity
    id: UUID = Field(default_factory=uuid4)
    item_type: str
    schema_version: str = "1.0"

    # Time (financial mandate)
    as_of: datetime = Field(..., description="Information cutoff time.")
    available_at: datetime | None = Field(
        default=None,
        description="Time the source became observable to researchers.",
    )
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

    # ── Convenience helpers ────────────────────────────────────────────
    # These are shared by every shape (Flatten / Tree / Graph) because they
    # operate on `BaseKnowledge` metadata, not domain payload.

    def is_extracted(self) -> bool:
        """True iff the item came from an LLM flow (vs hand-curated)."""
        return self.extraction is not None

    def freshness(self, now: datetime | None = None) -> timedelta:
        """Time elapsed since ``as_of``. Defaults ``now`` to ``utcnow()``."""
        reference = now if now is not None else datetime.now(timezone.utc)
        return reference - self.as_of

    def with_tags(self, *new_tags: str) -> Self:
        """Return a copy with extra tags appended (frozen-friendly).

        Duplicates are skipped so the operation is idempotent.
        """
        merged = list(self.tags)
        for t in new_tags:
            if t not in merged:
                merged.append(t)
        return self.model_copy(update={"tags": merged})
