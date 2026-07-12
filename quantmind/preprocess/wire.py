"""Provider-pluggable ingestion for public press-release wire feeds."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

import httpx

from quantmind.preprocess.fetch._types import Fetched
from quantmind.preprocess.fetch.http import (
    FetchAttemptsExhausted,
    FetchPolicy,
    HttpFetcher,
)
from quantmind.preprocess.fetch.rss import FeedItem, parse_feed
from quantmind.preprocess.news import (
    BodySource,
    NewsTickerHint,
    canonicalize_source_url,
    feed_item_to_news_document,
    news_document_from_fetched,
    preprocess_news_document,
)

WireFailureStage = Literal[
    "feed_fetch",
    "feed_parse",
    "article_fetch",
    "article_parse",
    "document_build",
]


@dataclass(frozen=True, slots=True)
class WireItemMapping:
    """Provider-normalized identity and metadata for one feed item."""

    payload_id: str | None
    canonical_url: str | None
    title: str
    published_at: datetime | None


class WireProvider(Protocol):
    """Adapter contract for provider-specific feed item behavior."""

    @property
    def name(self) -> str:
        """Stable provider identifier."""
        ...

    @property
    def publisher(self) -> str:
        """Human-readable publisher name."""
        ...

    @property
    def body_source(self) -> BodySource:
        """Whether cleaned body text comes from feed or article HTML."""
        ...

    def map_item(self, item: FeedItem) -> WireItemMapping:
        """Map one provider feed item into common identity fields."""
        ...


@dataclass(frozen=True, slots=True)
class _StandardWireProvider:
    name: str
    publisher: str
    body_source: BodySource

    def map_item(self, item: FeedItem) -> WireItemMapping:
        url = canonicalize_source_url(item.url) if item.url else None
        return WireItemMapping(
            payload_id=(item.id or "").strip() or None,
            canonical_url=url,
            title=item.title,
            published_at=item.published_at,
        )


PR_NEWSWIRE: WireProvider = _StandardWireProvider(
    name="pr-newswire",
    publisher="PR Newswire",
    body_source="article",
)


@dataclass(frozen=True, slots=True)
class WireFeedConfig:
    """Public feed URLs, provider adapter, and shared HTTP policy."""

    provider: WireProvider
    feed_urls: tuple[str, ...]
    fetch_policy: FetchPolicy = field(default_factory=FetchPolicy)

    def __post_init__(self) -> None:
        """Reject incomplete configs before network calls begin."""
        if not self.feed_urls:
            raise ValueError("WireFeedConfig requires at least one feed URL")
        if any(not url.strip() for url in self.feed_urls):
            raise ValueError("feed URLs must not be empty")


@dataclass(frozen=True, slots=True)
class RawWireArtifact:
    """Raw bytes and fetch metadata retained for replay and re-cleaning."""

    bytes: bytes
    content_hash: str
    content_type: str
    source_url: str | None
    resolved_url: str | None
    status_code: int | None
    headers: dict[str, str] = field(default_factory=dict)
    fetched_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class WireDocument:
    """Replayable raw wire evidence paired with cleaned markdown."""

    provider: str
    identity: str
    cleaned_markdown: str
    content_hash: str
    raw_feed_entry: RawWireArtifact
    raw_article: RawWireArtifact | None = None
    payload_id: str | None = None
    canonical_url: str | None = None
    title: str | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    ticker_hints: tuple[NewsTickerHint, ...] = ()


@dataclass(frozen=True, slots=True)
class WireFetchFailure:
    """Lightweight record for one feed or item that could not be processed."""

    provider: str
    stage: WireFailureStage
    source_url: str
    item_id: str | None
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class WireFetchResult:
    """Successful wire documents plus non-fatal per-item failures."""

    documents: tuple[WireDocument, ...] = ()
    failures: tuple[WireFetchFailure, ...] = ()

    @property
    def success_count(self) -> int:
        """Number of documents produced by this call."""
        return len(self.documents)

    @property
    def failure_count(self) -> int:
        """Number of feed or item failures recorded by this call."""
        return len(self.failures)


async def fetch_wire_documents(config: WireFeedConfig) -> WireFetchResult:
    """Fetch current feed items and convert them into wire documents.

    Independent feed and item failures are recorded while remaining inputs
    continue. Configuration errors still raise before any network request.
    """
    documents: list[WireDocument] = []
    failures: list[WireFetchFailure] = []
    seen: set[str] = set()
    async with HttpFetcher(policy=config.fetch_policy) as fetcher:
        for feed_url in config.feed_urls:
            try:
                fetched_feed = await fetcher.fetch_url(
                    feed_url,
                    max_bytes=5_000_000,
                )
            except Exception as exc:
                failures.append(
                    _failure(
                        config.provider,
                        "feed_fetch",
                        feed_url,
                        None,
                        exc,
                    )
                )
                continue

            try:
                feed = parse_feed(
                    fetched_feed.bytes,
                    feed_url=feed_url,
                    content_type=fetched_feed.content_type,
                    headers=fetched_feed.headers,
                    fetched=fetched_feed,
                )
            except Exception as exc:
                failures.append(
                    _failure(
                        config.provider,
                        "feed_parse",
                        feed_url,
                        None,
                        exc,
                    )
                )
                continue

            for item in feed.items:
                try:
                    mapping = config.provider.map_item(item)
                    identity = build_wire_identity(
                        provider=config.provider.name,
                        payload_id=mapping.payload_id,
                        canonical_url=mapping.canonical_url,
                    )
                except Exception as exc:
                    failures.append(
                        _failure(
                            config.provider,
                            "document_build",
                            item.url or feed_url,
                            item.id,
                            exc,
                        )
                    )
                    continue

                if identity in seen:
                    continue
                seen.add(identity)

                document = await _build_document(
                    provider=config.provider,
                    item=item,
                    mapping=mapping,
                    identity=identity,
                    fetched_feed=fetched_feed,
                    fetcher=fetcher,
                    failures=failures,
                )
                if document is not None:
                    documents.append(document)

    return WireFetchResult(
        documents=tuple(documents),
        failures=tuple(failures),
    )


def build_wire_identity(
    *,
    provider: str,
    payload_id: str | None,
    canonical_url: str | None,
) -> str:
    """Build stable identity from provider, payload ID, and canonical URL."""
    normalized_provider = provider.strip().lower()
    normalized_payload = (payload_id or "").strip()
    normalized_url = (
        canonicalize_source_url(canonical_url) if canonical_url else ""
    )
    if not normalized_provider:
        raise ValueError("wire identity requires a provider")
    if not normalized_payload and not normalized_url:
        raise ValueError("wire identity requires a payload ID or URL")
    raw_identity = "\x1f".join(
        (normalized_provider, normalized_payload, normalized_url)
    )
    digest = hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()
    return f"wire:{normalized_provider}:{digest}"


async def _build_document(
    *,
    provider: WireProvider,
    item: FeedItem,
    mapping: WireItemMapping,
    identity: str,
    fetched_feed: Fetched,
    fetcher: HttpFetcher,
    failures: list[WireFetchFailure],
) -> WireDocument | None:
    raw_article: Fetched | None = None
    if provider.body_source == "article":
        article_url = mapping.canonical_url
        if not article_url:
            failures.append(
                _failure(
                    provider,
                    "article_fetch",
                    item.source_feed_url or "",
                    item.id,
                    ValueError("article body source requires an item URL"),
                )
            )
            return None
        try:
            raw_article = await fetcher.fetch_url(
                article_url,
                max_bytes=10_000_000,
            )
        except Exception as exc:
            failures.append(
                _failure(
                    provider,
                    "article_fetch",
                    article_url,
                    item.id,
                    exc,
                )
            )
            return None

    try:
        if raw_article is None:
            raw_news = await feed_item_to_news_document(
                item,
                publisher=provider.publisher,
                body_source="feed",
            )
        else:
            raw_news = await news_document_from_fetched(
                raw_article,
                title=mapping.title,
                publisher=provider.publisher,
                published_at=mapping.published_at,
                payload_id=mapping.payload_id,
            )
        candidate = preprocess_news_document(raw_news)
    except Exception as exc:
        failures.append(
            _failure(
                provider,
                "article_parse"
                if raw_article is not None
                else "document_build",
                item.url or item.source_feed_url or "",
                item.id,
                exc,
            )
        )
        return None

    return WireDocument(
        provider=provider.name,
        identity=identity,
        cleaned_markdown=candidate.body_text,
        content_hash=candidate.content_hash,
        raw_feed_entry=_feed_entry_artifact(item, fetched_feed),
        raw_article=(
            _artifact_from_fetched(raw_article)
            if raw_article is not None
            else None
        ),
        payload_id=mapping.payload_id,
        canonical_url=mapping.canonical_url,
        title=candidate.title,
        publisher=candidate.publisher,
        published_at=candidate.published_at,
        ticker_hints=candidate.ticker_hints,
    )


def _feed_entry_artifact(
    item: FeedItem,
    fetched_feed: Fetched,
) -> RawWireArtifact:
    raw_xml = item.raw_xml
    return RawWireArtifact(
        bytes=raw_xml,
        content_hash=hashlib.sha256(raw_xml).hexdigest(),
        content_type="application/xml",
        source_url=fetched_feed.source_url,
        resolved_url=fetched_feed.resolved_url,
        status_code=fetched_feed.status_code,
        headers=dict(fetched_feed.headers),
        fetched_at=fetched_feed.fetched_at,
    )


def _artifact_from_fetched(fetched: Fetched) -> RawWireArtifact:
    return RawWireArtifact(
        bytes=fetched.bytes,
        content_hash=hashlib.sha256(fetched.bytes).hexdigest(),
        content_type=fetched.content_type,
        source_url=fetched.source_url,
        resolved_url=fetched.resolved_url,
        status_code=fetched.status_code,
        headers=dict(fetched.headers),
        fetched_at=fetched.fetched_at,
    )


def _failure(
    provider: WireProvider,
    stage: WireFailureStage,
    source_url: str,
    item_id: str | None,
    error: Exception,
) -> WireFetchFailure:
    return WireFetchFailure(
        provider=provider.name,
        stage=stage,
        source_url=source_url,
        item_id=item_id,
        error_type=_error_type(error),
        message=str(error),
    )


def _error_type(error: Exception) -> str:
    if isinstance(error, FetchAttemptsExhausted):
        return "retry_exhausted"
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    if isinstance(error, httpx.TransportError):
        return "network"
    if isinstance(error, httpx.HTTPStatusError):
        return "http_status"
    if isinstance(error, ValueError):
        return "invalid_content"
    return "unexpected"
