"""Source-faithful news collection values, exported by ``preprocess``.

These records carry acquisition evidence and status. They are intentionally
separate from semantic ``quantmind.knowledge.News`` values.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

NewsFailureStage = Literal[
    "discovery_fetch",
    "discovery_parse",
    "article_fetch",
    "article_parse",
]


@dataclass(frozen=True, slots=True)
class NewsTickerHint:
    """Ticker hint extracted before instrument resolution."""

    symbol: str
    exchange: str | None = None
    source: str = "exchange_code"
    confidence: float = 1.0
    raw: str | None = None


@dataclass(frozen=True, slots=True)
class NewsArtifact:
    """Raw evidence and fetch metadata retained for replay or auditing."""

    bytes: bytes | None
    content_hash: str
    content_type: str
    source_url: str | None
    resolved_url: str | None
    status_code: int | None
    headers: dict[str, str] = field(default_factory=dict)
    fetched_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class NewsDocument:
    """One source observation with cleaned text and its raw evidence."""

    source: str
    identity: str
    cleaned_markdown: str
    content_hash: str
    discovery_artifact: NewsArtifact
    article_artifact: NewsArtifact
    payload_id: str | None = None
    canonical_url: str | None = None
    title: str | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    ticker_hints: tuple[NewsTickerHint, ...] = ()


@dataclass(frozen=True, slots=True)
class NewsFailure:
    """Lightweight record for one recoverable collection failure."""

    source: str
    stage: NewsFailureStage
    source_url: str
    item_id: str | None
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class NewsBatch:
    """Observed documents, failures, and discovery-coverage status."""

    documents: tuple[NewsDocument, ...] = ()
    failures: tuple[NewsFailure, ...] = ()
    observed_count: int = 0
    complete: bool = False

    @property
    def success_count(self) -> int:
        """Number of successfully collected observations."""
        return len(self.documents)

    @property
    def failure_count(self) -> int:
        """Number of recorded recoverable failures."""
        return len(self.failures)
