"""Source-agnostic news preprocessing and candidate normalization."""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from quantmind.preprocess.clean import (
    collapse_whitespace,
    dedupe_lines,
    normalize_unicode,
)
from quantmind.preprocess.fetch._types import Fetched
from quantmind.preprocess.fetch.http import HttpFetcher, fetch_url
from quantmind.preprocess.fetch.rss import FeedItem
from quantmind.preprocess.format.html import html_to_markdown
from quantmind.preprocess.time import to_utc

NewsSourceType = Literal[
    "company_8k",
    "press_release",
    "publisher_news",
    "regulatory_news",
]
BodySource = Literal["feed", "article"]
NewsFailureStage = Literal[
    "discovery_fetch",
    "discovery_parse",
    "article_fetch",
    "article_parse",
]

_SOURCE_PREFIX: dict[str, str] = {
    "company_8k": "sec",
    "press_release": "wire",
    "publisher_news": "publisher",
    "regulatory_news": "regulatory",
}

_TRACKING_QUERY_PARAMS = {
    "fbclid",
    "feedref",
    "gclid",
    "mc_cid",
    "mc_eid",
    "spm",
}

_EXCHANGE_TICKER_RE = re.compile(
    r"(?:\(|\b)"
    r"(NASDAQ|NYSE(?:\s+American|\s+MKT|\s+Arca)?|AMEX|OTCQX|OTCQB|OTC|CBOE)"
    r"\s*:\s*"
    r"([A-Z][A-Z0-9.-]{0,9})"
    r"(?:\)|\b)",
    re.IGNORECASE,
)
_EMAIL_PROTECTION_LINK_RE = re.compile(
    r"\[\[email protected]\]\(/cdn-cgi/l/email-protection#[^)]+\)"
)

_EXCHANGE_NAMES: dict[str, str] = {
    "NASDAQ": "NASDAQ",
    "NYSE": "NYSE",
    "NYSE AMERICAN": "NYSE American",
    "NYSE MKT": "NYSE American",
    "NYSE ARCA": "NYSE Arca",
    "AMEX": "NYSE American",
    "OTCQX": "OTCQX",
    "OTCQB": "OTCQB",
    "OTC": "OTC",
    "CBOE": "CBOE",
}


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


@dataclass(frozen=True, slots=True)
class RawNewsDocument:
    """Raw news document ready for source-agnostic preprocessing."""

    body_text: str
    source_type: NewsSourceType = "press_release"
    source_url: str | None = None
    title: str | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    payload_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NewsCandidate:
    """Normalised candidate document consumed by a news processing flow."""

    body_text: str
    content_hash: str
    source_type: NewsSourceType
    identity: str
    source_url: str | None = None
    title: str | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    ticker_hints: tuple[NewsTickerHint, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


async def fetch_news_document(
    url: str,
    *,
    source_type: NewsSourceType = "press_release",
    title: str | None = None,
    publisher: str | None = None,
    published_at: datetime | None = None,
    payload_id: str | None = None,
    timeout: float = 30.0,
    max_bytes: int = 10_000_000,
    fetcher: HttpFetcher | None = None,
) -> RawNewsDocument:
    """Fetch one news/press-release URL and extract plain markdown text.

    Args:
        url: Article or press-release URL.
        source_type: Normalised source family.
        title: Optional title supplied by a feed or caller.
        publisher: Optional source publisher.
        published_at: Optional publication timestamp.
        payload_id: Optional upstream feed/item id.
        timeout: Per-request timeout in seconds.
        max_bytes: Hard ceiling on response body size.
        fetcher: Optional shared HTTP fetcher and host-rate state.

    Returns:
        Raw document with extracted markdown text and caller metadata.

    Raises:
        ValueError: If the content type cannot be converted to text.
        httpx.HTTPError: For network / status / timeout failures.
    """
    if fetcher is None:
        fetched = await fetch_url(url, timeout=timeout, max_bytes=max_bytes)
    else:
        fetched = await fetcher.fetch_url(
            url,
            timeout=timeout,
            max_bytes=max_bytes,
        )
    return await news_document_from_fetched(
        fetched,
        source_type=source_type,
        title=title,
        publisher=publisher,
        published_at=published_at,
        payload_id=payload_id,
    )


async def news_document_from_fetched(
    fetched: Fetched,
    *,
    source_type: NewsSourceType = "press_release",
    title: str | None = None,
    publisher: str | None = None,
    published_at: datetime | None = None,
    payload_id: str | None = None,
) -> RawNewsDocument:
    """Extract a raw news document from already-fetched evidence."""
    body_text = await _text_from_fetched(fetched)
    return RawNewsDocument(
        body_text=body_text,
        source_type=source_type,
        source_url=fetched.resolved_url or fetched.source_url,
        title=title,
        publisher=publisher,
        published_at=published_at,
        payload_id=payload_id,
        metadata={
            "body_source": "article",
            "content_type": fetched.content_type,
        },
    )


async def preprocess_news_url(
    url: str,
    *,
    source_type: NewsSourceType = "press_release",
    title: str | None = None,
    publisher: str | None = None,
    published_at: datetime | None = None,
    payload_id: str | None = None,
    timeout: float = 30.0,
    max_bytes: int = 10_000_000,
) -> NewsCandidate:
    """Fetch and normalise one news/press-release URL."""
    raw = await fetch_news_document(
        url,
        source_type=source_type,
        title=title,
        publisher=publisher,
        published_at=published_at,
        payload_id=payload_id,
        timeout=timeout,
        max_bytes=max_bytes,
    )
    return preprocess_news_document(raw)


async def feed_item_to_news_document(
    item: FeedItem,
    *,
    source_type: NewsSourceType = "press_release",
    publisher: str | None = None,
    body_source: BodySource = "feed",
    fetcher: HttpFetcher | None = None,
) -> RawNewsDocument:
    """Convert one parsed RSS/Atom item into a raw news document.

    ``body_source="feed"`` trusts the RSS/Atom body. ``"article"`` always
    follows the item URL, even when the feed contains a non-empty teaser.
    """
    if body_source == "article":
        if not item.url:
            raise ValueError("article body_source requires an item URL")
        raw = await fetch_news_document(
            item.url,
            source_type=source_type,
            title=item.title,
            publisher=publisher,
            published_at=item.published_at,
            payload_id=item.id,
            fetcher=fetcher,
        )
        return RawNewsDocument(
            body_text=raw.body_text,
            source_type=raw.source_type,
            source_url=raw.source_url,
            title=raw.title,
            publisher=raw.publisher,
            published_at=raw.published_at,
            payload_id=raw.payload_id,
            metadata={
                **raw.metadata,
                **{
                    key: value
                    for key, value in {
                        "source_feed_url": item.source_feed_url,
                        **item.raw,
                    }.items()
                    if value
                },
            },
        )

    body_text = await _html_or_text_to_markdown(
        item.content_html or item.summary_html or ""
    )
    return RawNewsDocument(
        body_text=body_text,
        source_type=source_type,
        source_url=item.url,
        title=item.title,
        publisher=publisher,
        published_at=item.published_at,
        payload_id=item.id,
        metadata={
            key: value
            for key, value in {
                "body_source": "feed",
                "source_feed_url": item.source_feed_url,
                **item.raw,
            }.items()
            if value
        },
    )


async def preprocess_feed_item(
    item: FeedItem,
    *,
    source_type: NewsSourceType = "press_release",
    publisher: str | None = None,
    body_source: BodySource = "feed",
    fetcher: HttpFetcher | None = None,
) -> NewsCandidate:
    """Convert and normalise one parsed RSS/Atom item."""
    raw = await feed_item_to_news_document(
        item,
        source_type=source_type,
        publisher=publisher,
        body_source=body_source,
        fetcher=fetcher,
    )
    return preprocess_news_document(raw)


def preprocess_news_document(raw: RawNewsDocument) -> NewsCandidate:
    """Normalise a raw news document into the shared candidate contract.

    The output carries stable source identity, cleaned body text, a content
    hash, provenance, timestamp, and deterministic ticker hints.
    """
    body_text = normalize_news_text(raw.body_text)
    if not body_text:
        raise ValueError("news document body is empty after preprocessing")

    source_url = (
        canonicalize_source_url(raw.source_url) if raw.source_url else None
    )
    title = normalize_news_text(raw.title or "") or None
    published_at = to_utc(raw.published_at) if raw.published_at else None
    ticker_scan = "\n".join(part for part in (title, body_text) if part)

    return NewsCandidate(
        body_text=body_text,
        content_hash=news_content_hash(body_text),
        source_type=raw.source_type,
        identity=build_news_identity(
            source_type=raw.source_type,
            source_url=source_url,
            payload_id=raw.payload_id,
        ),
        source_url=source_url,
        title=title,
        publisher=raw.publisher,
        published_at=published_at,
        ticker_hints=extract_exchange_ticker_hints(ticker_scan),
        metadata=dict(raw.metadata),
    )


def normalize_news_text(text: str) -> str:
    """Apply QuantMind's canonical text cleanup order for news bodies."""
    normalized = normalize_unicode(text)
    normalized = _EMAIL_PROTECTION_LINK_RE.sub("[email protected]", normalized)
    return collapse_whitespace(dedupe_lines(normalized))


def news_content_hash(text: str) -> str:
    """Return the sha256 hash for an already-normalised news body."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_news_identity(
    *,
    source_type: NewsSourceType,
    source_url: str | None = None,
    payload_id: str | None = None,
) -> str:
    """Build a deterministic source-document identity.

    Press-release/wire rows use the payload id when available and otherwise
    fall back to the canonical source URL. The identity is hashed so callers do
    not accidentally depend on upstream id formatting.
    """
    identity = (payload_id or "").strip()
    if not identity and source_url:
        identity = canonicalize_source_url(source_url)
    if not identity:
        raise ValueError("news identity requires payload_id or source_url")
    prefix = _SOURCE_PREFIX[source_type]
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def build_sec_news_identity(
    *,
    accession_number: str,
    section_key: str,
) -> str:
    """Build a stable 8-K EX-99.x news identity."""
    accession = accession_number.strip()
    section = section_key.strip().lower()
    if not accession or not section:
        raise ValueError("SEC news identity requires accession and section")
    return f"sec:{accession}:{section}"


def canonicalize_source_url(url: str) -> str:
    """Normalise source URLs for stable source identity."""
    parsed = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in _TRACKING_QUERY_PARAMS
    ]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            urlencode(sorted(query), doseq=True),
            "",
        )
    )


def extract_exchange_ticker_hints(text: str) -> tuple[NewsTickerHint, ...]:
    """Extract exchange-qualified ticker mentions from PR-style text.

    Examples matched include ``(NASDAQ: NVDA)`` and ``NYSE: IBM``. The result is
    only a hint; downstream instrument resolution should still validate it.
    """
    hints: list[NewsTickerHint] = []
    seen: set[tuple[str, str | None]] = set()
    for match in _EXCHANGE_TICKER_RE.finditer(text):
        raw_exchange = " ".join(match.group(1).upper().split())
        exchange = _EXCHANGE_NAMES.get(raw_exchange, raw_exchange)
        symbol = match.group(2).upper()
        key = (symbol, exchange)
        if key in seen:
            continue
        seen.add(key)
        hints.append(
            NewsTickerHint(
                symbol=symbol,
                exchange=exchange,
                raw=match.group(0).strip(),
            )
        )
    return tuple(hints)


async def _text_from_fetched(raw: Fetched) -> str:
    ct = (raw.content_type or "").lower()
    text = raw.bytes.decode("utf-8", errors="replace")
    if ct.startswith("text/html") or ct.startswith("application/xhtml+xml"):
        return await html_to_markdown(text)
    if (
        ct.startswith("text/plain")
        or ct.startswith("text/markdown")
        or ct.startswith("application/xml")
        or ct.startswith("text/xml")
    ):
        return text
    raise ValueError(f"Unsupported content-type for news input: {ct!r}")


async def _html_or_text_to_markdown(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if "<" not in text or ">" not in text:
        return text
    markdown = await html_to_markdown(text)
    return markdown or text
