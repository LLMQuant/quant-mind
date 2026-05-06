"""Frozen dataclass output types for the fetch.* layer.

`Fetched` is the common shape every `fetch_*` function returns. `RawPaper`
extends it with arxiv-specific metadata pulled from the arxiv API. Flow
layer (PR5) maps these into typed `quantmind.knowledge` schemas.

These are intentionally dataclasses (not Pydantic) — fetch is internal
plumbing, not an LLM boundary, so we want zero validation overhead and
hashable value types.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Fetched:
    """Common output shape of every fetch.* function."""

    bytes: bytes
    """Raw payload pulled from the source."""

    content_type: str
    """MIME type, e.g. ``application/pdf`` / ``text/html`` / ``text/markdown``."""

    source_url: str | None = None
    """Origin URL when applicable; ``None`` for local-file fetches."""

    headers: dict[str, str] = field(default_factory=dict)
    """Selected response headers (HTTP fetches only); empty dict otherwise."""


@dataclass(frozen=True, slots=True)
class RawPaper(Fetched):
    """Arxiv-specific fetch result with metadata attached.

    All metadata fields default to empty/None so callers can construct
    instances incrementally during testing without supplying everything.
    """

    arxiv_id: str = ""
    title: str | None = None
    authors: tuple[str, ...] = ()
    abstract: str | None = None
    published_at: datetime | None = None
    primary_category: str | None = None
    categories: tuple[str, ...] = ()
