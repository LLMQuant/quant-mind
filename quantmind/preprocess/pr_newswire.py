"""Internal PR Newswire source implementation for ``collect_news``."""

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlencode, urljoin
from zoneinfo import ZoneInfo

import httpx

from quantmind.preprocess._news_types import (
    NewsArtifact,
    NewsBatch,
    NewsDocument,
    NewsFailure,
    NewsFailureStage,
)
from quantmind.preprocess.fetch._types import Fetched
from quantmind.preprocess.fetch.http import (
    FetchAttemptsExhausted,
    FetchPolicy,
    HttpFetcher,
)
from quantmind.preprocess.news import (
    canonicalize_source_url,
    news_document_from_fetched,
    preprocess_news_document,
)

_SOURCE = "pr-newswire"
_PUBLISHER = "PR Newswire"
_BASE_URL = "https://www.prnewswire.com"
_LISTING_URL = f"{_BASE_URL}/news-releases/news-releases-list/"
_EASTERN = ZoneInfo("America/New_York")
_PAGE_SIZE = 100
_MAX_PAGES = 20
_CACHE_BUST_ATTEMPTS = 8
_ARTICLE_MAX_BYTES = 10_000_000
_LISTING_MAX_BYTES = 5_000_000
_DEFAULT_FETCH_POLICY = FetchPolicy(
    max_attempts=3,
    backoff_base_seconds=0.5,
    backoff_max_seconds=5.0,
    jitter_seconds=0.1,
    max_concurrency_per_host=2,
    min_interval_seconds=0.1,
)

_SHORT_TIME_RE = re.compile(
    r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})\s+E(?:S|D)?T$",
    re.IGNORECASE,
)
_FULL_TIME_RE = re.compile(
    r"^(?P<month>[A-Za-z]{3})\s+(?P<day>\d{1,2}),\s+"
    r"(?P<year>\d{4}),\s+(?P<hour>\d{1,2}):"
    r"(?P<minute>\d{2})\s+E(?:S|D)?T$",
    re.IGNORECASE,
)
_MONTHS = {
    month: number
    for number, month in enumerate(
        (
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ),
        start=1,
    )
}
_PAYLOAD_ID_RE = re.compile(r"-(?P<id>\d+)\.html$")


@dataclass(frozen=True, slots=True)
class PRNewswireObservation:
    """One public listing row inside the requested time window."""

    identity: str
    payload_id: str | None
    canonical_url: str
    title: str
    published_at: datetime
    discovery_artifact: NewsArtifact


@dataclass(frozen=True, slots=True)
class PRNewswireDiscovery:
    """Listing observations and evidence that window discovery completed."""

    observations: tuple[PRNewswireObservation, ...] = ()
    failures: tuple[NewsFailure, ...] = ()
    page_count: int = 0
    complete: bool = False

    @property
    def observed_count(self) -> int:
        """Number of listing rows retained inside the requested window."""
        return len(self.observations)


@dataclass(frozen=True, slots=True)
class _ParsedListingRow:
    href: str | None
    title: str
    timestamp: str
    raw_html: bytes


@dataclass(frozen=True, slots=True)
class _ParsedListingPage:
    page_date: date
    rows: tuple[_ParsedListingRow, ...]


class _ListingParser(HTMLParser):
    """Extract only PR Newswire listing cards with no optional dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.page_date_text: str | None = None
        self.rows: list[_ParsedListingRow] = []
        self._row_div_depth = 0
        self._raw_parts: list[str] = []
        self._href: str | None = None
        self._title_parts: list[str] = []
        self._timestamp_parts: list[str] = []
        self._in_heading = False
        self._in_timestamp = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if (
            not self._row_div_depth
            and tag == "div"
            and {"row", "newsCards"}.issubset(classes)
        ):
            self._start_row()

        if not self._row_div_depth:
            if tag == "input" and attributes.get("id") == "date":
                self.page_date_text = attributes.get("value")
            return

        start_text = self.get_starttag_text() or f"<{tag}>"
        self._raw_parts.append(start_text)
        if tag == "div" and len(self._raw_parts) > 1:
            self._row_div_depth += 1
        if tag == "a" and "newsreleaseconsolidatelink" in classes:
            self._href = self._href or attributes.get("href")
        if tag == "h3":
            self._in_heading = True
        if tag == "small" and self._in_heading:
            self._in_timestamp = True

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if not self._row_div_depth:
            if tag == "input" and attributes.get("id") == "date":
                self.page_date_text = attributes.get("value")
            return
        self._raw_parts.append(self.get_starttag_text() or f"<{tag}/>")

    def handle_endtag(self, tag: str) -> None:
        if not self._row_div_depth:
            return
        self._raw_parts.append(f"</{tag}>")
        if tag == "small":
            self._in_timestamp = False
        if tag == "h3":
            self._in_heading = False
            self._in_timestamp = False
        if tag == "div":
            self._row_div_depth -= 1
            if not self._row_div_depth:
                self._finish_row()

    def handle_data(self, data: str) -> None:
        if not self._row_div_depth:
            return
        self._raw_parts.append(data)
        self._append_heading_text(data)

    def handle_entityref(self, name: str) -> None:
        self._handle_reference(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._handle_reference(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        if self._row_div_depth:
            self._raw_parts.append(f"<!--{data}-->")

    def _start_row(self) -> None:
        self._row_div_depth = 1
        self._raw_parts = []
        self._href = None
        self._title_parts = []
        self._timestamp_parts = []
        self._in_heading = False
        self._in_timestamp = False

    def _finish_row(self) -> None:
        self.rows.append(
            _ParsedListingRow(
                href=self._href,
                title=_normalize_space(unescape("".join(self._title_parts))),
                timestamp=_normalize_space(
                    unescape("".join(self._timestamp_parts))
                ),
                raw_html="".join(self._raw_parts).encode("utf-8"),
            )
        )
        self._raw_parts = []

    def _handle_reference(self, reference: str) -> None:
        if not self._row_div_depth:
            return
        self._raw_parts.append(reference)
        self._append_heading_text(reference)

    def _append_heading_text(self, text: str) -> None:
        if not self._in_heading:
            return
        if self._in_timestamp:
            self._timestamp_parts.append(text)
        else:
            self._title_parts.append(text)


async def _discover_pr_newswire(
    *,
    start: datetime,
    end: datetime,
) -> PRNewswireDiscovery:
    """Discover every public PR Newswire listing row in ``[start, end)``."""
    start, end = _validate_window(start, end)
    async with HttpFetcher(policy=_DEFAULT_FETCH_POLICY) as fetcher:
        return await _discover_with_fetcher(
            start=start,
            end=end,
            fetcher=fetcher,
        )


async def _collect_pr_newswire(
    *,
    start: datetime,
    end: datetime,
    retain_raw_html: bool,
) -> NewsBatch:
    """Discover and collect PR Newswire articles for ``[start, end)``."""
    start, end = _validate_window(start, end)
    async with HttpFetcher(policy=_DEFAULT_FETCH_POLICY) as fetcher:
        discovery = await _discover_with_fetcher(
            start=start,
            end=end,
            fetcher=fetcher,
        )
        outcomes = await asyncio.gather(
            *(
                _collect_observation(
                    observation,
                    fetcher=fetcher,
                    retain_raw_html=retain_raw_html,
                )
                for observation in discovery.observations
            )
        )

    documents = tuple(
        outcome for outcome in outcomes if isinstance(outcome, NewsDocument)
    )
    article_failures = tuple(
        outcome for outcome in outcomes if isinstance(outcome, NewsFailure)
    )
    return NewsBatch(
        documents=documents,
        failures=discovery.failures + article_failures,
        observed_count=discovery.observed_count,
        complete=discovery.complete,
    )


async def _discover_with_fetcher(
    *,
    start: datetime,
    end: datetime,
    fetcher: HttpFetcher,
) -> PRNewswireDiscovery:
    observations: list[PRNewswireObservation] = []
    failures: list[NewsFailure] = []
    page_count = 0
    crossed_start = False
    anchor = _ceil_to_hour(end).astimezone(_EASTERN)
    prior_page: tuple[tuple[str | None, str], ...] | None = None
    last_published_at: datetime | None = None

    for page_number in range(1, _MAX_PAGES + 1):
        try:
            fetched = await _fetch_listing_page(
                fetcher,
                anchor=anchor,
                page_number=page_number,
            )
        except Exception as exc:
            failures.append(
                _failure(
                    stage="discovery_fetch",
                    source_url=_listing_url(
                        anchor=anchor,
                        page_number=page_number,
                    ),
                    item_id=None,
                    error=exc,
                )
            )
            break

        page_count += 1
        try:
            parsed = _parse_listing_page(fetched, fallback_date=anchor.date())
        except Exception as exc:
            failures.append(
                _failure(
                    stage="discovery_parse",
                    source_url=(
                        fetched.resolved_url
                        or fetched.source_url
                        or _LISTING_URL
                    ),
                    item_id=None,
                    error=exc,
                )
            )
            break

        fingerprint = tuple((row.href, row.timestamp) for row in parsed.rows)
        if not fingerprint:
            failures.append(
                _failure(
                    stage="discovery_parse",
                    source_url=(
                        fetched.resolved_url
                        or fetched.source_url
                        or _LISTING_URL
                    ),
                    item_id=None,
                    error=ValueError("PR Newswire listing contained no rows"),
                )
            )
            break
        if fingerprint == prior_page:
            failures.append(
                _failure(
                    stage="discovery_parse",
                    source_url=(
                        fetched.resolved_url
                        or fetched.source_url
                        or _LISTING_URL
                    ),
                    item_id=None,
                    error=ValueError("PR Newswire pagination did not advance"),
                )
            )
            break
        prior_page = fingerprint

        for row in parsed.rows:
            try:
                published_at = _parse_listing_timestamp(
                    row.timestamp,
                    page_date=parsed.page_date,
                    previous=last_published_at,
                )
                last_published_at = published_at
                observation = _observation_from_row(
                    row,
                    published_at=published_at,
                    fetched_page=fetched,
                )
            except Exception as exc:
                failures.append(
                    _failure(
                        stage="discovery_parse",
                        source_url=(
                            fetched.resolved_url
                            or fetched.source_url
                            or _LISTING_URL
                        ),
                        item_id=row.href,
                        error=exc,
                    )
                )
                continue

            if start <= published_at < end:
                observations.append(observation)
            if published_at < start:
                crossed_start = True

        if crossed_start:
            break
    else:
        failures.append(
            _failure(
                stage="discovery_parse",
                source_url=_LISTING_URL,
                item_id=None,
                error=ValueError(
                    f"PR Newswire discovery exceeded {_MAX_PAGES} pages"
                ),
            )
        )

    return PRNewswireDiscovery(
        observations=tuple(observations),
        failures=tuple(failures),
        page_count=page_count,
        complete=crossed_start and not failures,
    )


async def _fetch_listing_page(
    fetcher: HttpFetcher,
    *,
    anchor: datetime,
    page_number: int,
) -> Fetched:
    last_error: Exception | None = None
    for _ in range(_CACHE_BUST_ATTEMPTS):
        url = _listing_url(
            anchor=anchor,
            page_number=page_number,
            cache_bust=str(time.time_ns()),
        )
        try:
            return await fetcher.fetch_url(url, max_bytes=_LISTING_MAX_BYTES)
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code != 404:
                raise
    assert last_error is not None
    raise last_error


async def _collect_observation(
    observation: PRNewswireObservation,
    *,
    fetcher: HttpFetcher,
    retain_raw_html: bool,
) -> NewsDocument | NewsFailure:
    try:
        fetched = await fetcher.fetch_url(
            observation.canonical_url,
            max_bytes=_ARTICLE_MAX_BYTES,
        )
    except Exception as exc:
        return _failure(
            stage="article_fetch",
            source_url=observation.canonical_url,
            item_id=observation.payload_id,
            error=exc,
        )

    try:
        raw = await news_document_from_fetched(
            fetched,
            title=observation.title,
            publisher=_PUBLISHER,
            published_at=observation.published_at,
            payload_id=observation.payload_id,
        )
        candidate = preprocess_news_document(raw)
    except Exception as exc:
        return _failure(
            stage="article_parse",
            source_url=observation.canonical_url,
            item_id=observation.payload_id,
            error=exc,
        )

    return NewsDocument(
        source=_SOURCE,
        identity=observation.identity,
        cleaned_markdown=candidate.body_text,
        content_hash=candidate.content_hash,
        discovery_artifact=observation.discovery_artifact,
        article_artifact=_artifact_from_fetched(
            fetched,
            retain_bytes=retain_raw_html,
        ),
        payload_id=observation.payload_id,
        canonical_url=observation.canonical_url,
        title=candidate.title,
        publisher=candidate.publisher,
        published_at=candidate.published_at,
        ticker_hints=candidate.ticker_hints,
    )


def _parse_listing_page(
    fetched: Fetched,
    *,
    fallback_date: date,
) -> _ParsedListingPage:
    parser = _ListingParser()
    try:
        parser.feed(fetched.bytes.decode("utf-8"))
        parser.close()
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid PR Newswire listing HTML: {exc}") from exc

    page_date = fallback_date
    if parser.page_date_text:
        try:
            page_date = datetime.strptime(
                parser.page_date_text,
                "%m/%d/%Y",
            ).date()
        except ValueError as exc:
            raise ValueError(
                f"invalid PR Newswire page date: {parser.page_date_text!r}"
            ) from exc
    return _ParsedListingPage(page_date=page_date, rows=tuple(parser.rows))


def _observation_from_row(
    row: _ParsedListingRow,
    *,
    published_at: datetime,
    fetched_page: Fetched,
) -> PRNewswireObservation:
    if not row.href:
        raise ValueError("PR Newswire listing row has no release URL")
    if not row.title:
        raise ValueError("PR Newswire listing row has no title")

    canonical_url = canonicalize_source_url(urljoin(_BASE_URL, row.href))
    payload_match = _PAYLOAD_ID_RE.search(canonical_url)
    payload_id = payload_match.group("id") if payload_match else None
    identity = _build_identity(
        payload_id=payload_id,
        canonical_url=canonical_url,
    )
    return PRNewswireObservation(
        identity=identity,
        payload_id=payload_id,
        canonical_url=canonical_url,
        title=row.title,
        published_at=published_at,
        discovery_artifact=NewsArtifact(
            bytes=row.raw_html,
            content_hash=hashlib.sha256(row.raw_html).hexdigest(),
            content_type="text/html",
            source_url=fetched_page.source_url,
            resolved_url=fetched_page.resolved_url,
            status_code=fetched_page.status_code,
            headers=dict(fetched_page.headers),
            fetched_at=fetched_page.fetched_at,
        ),
    )


def _parse_listing_timestamp(
    value: str,
    *,
    page_date: date,
    previous: datetime | None,
) -> datetime:
    previous_local = previous.astimezone(_EASTERN) if previous else None
    short_match = _SHORT_TIME_RE.fullmatch(value)
    if short_match:
        local_date = previous_local.date() if previous_local else page_date
        hour = int(short_match.group("hour"))
        minute = int(short_match.group("minute"))
    else:
        full_match = _FULL_TIME_RE.fullmatch(value)
        if not full_match:
            raise ValueError(f"invalid PR Newswire timestamp: {value!r}")
        month_name = full_match.group("month").lower()
        try:
            month = _MONTHS[month_name]
        except KeyError as exc:
            raise ValueError(
                f"invalid PR Newswire month: {month_name!r}"
            ) from exc
        local_date = date(
            int(full_match.group("year")),
            month,
            int(full_match.group("day")),
        )
        hour = int(full_match.group("hour"))
        minute = int(full_match.group("minute"))

    local = datetime.combine(
        local_date,
        datetime.min.time(),
        tzinfo=_EASTERN,
    ).replace(hour=hour, minute=minute)
    if (
        short_match
        and previous_local is not None
        and (hour, minute) > (previous_local.hour, previous_local.minute)
    ):
        local -= timedelta(days=1)
    return local.astimezone(timezone.utc)


def _listing_url(
    *,
    anchor: datetime,
    page_number: int,
    cache_bust: str | None = None,
) -> str:
    params = {
        "page": str(page_number),
        "pagesize": str(_PAGE_SIZE),
        "month": str(anchor.month),
        "day": str(anchor.day),
        "year": str(anchor.year),
        "hour": f"{anchor.hour:02d}",
    }
    if cache_bust is not None:
        params["_"] = cache_bust
    return f"{_LISTING_URL}?{urlencode(params)}"


def _ceil_to_hour(value: datetime) -> datetime:
    value = value.astimezone(timezone.utc)
    if value.minute or value.second or value.microsecond:
        return value.replace(minute=0, second=0, microsecond=0) + timedelta(
            hours=1
        )
    return value


def _validate_window(
    start: datetime,
    end: datetime,
) -> tuple[datetime, datetime]:
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("PR Newswire discovery requires an aware start")
    if end.tzinfo is None or end.utcoffset() is None:
        raise ValueError("PR Newswire discovery requires an aware end")
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if end <= start:
        raise ValueError("PR Newswire discovery requires end after start")
    return start, end


def _artifact_from_fetched(
    fetched: Fetched,
    *,
    retain_bytes: bool,
) -> NewsArtifact:
    return NewsArtifact(
        bytes=fetched.bytes if retain_bytes else None,
        content_hash=hashlib.sha256(fetched.bytes).hexdigest(),
        content_type=fetched.content_type,
        source_url=fetched.source_url,
        resolved_url=fetched.resolved_url,
        status_code=fetched.status_code,
        headers=dict(fetched.headers),
        fetched_at=fetched.fetched_at,
    )


def _build_identity(
    *,
    payload_id: str | None,
    canonical_url: str,
) -> str:
    raw = "\x1f".join((_SOURCE, payload_id or canonical_url))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"news:{_SOURCE}:{digest}"


def _failure(
    *,
    stage: NewsFailureStage,
    source_url: str,
    item_id: str | None,
    error: Exception,
) -> NewsFailure:
    return NewsFailure(
        source=_SOURCE,
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


def _normalize_space(value: str) -> str:
    return " ".join(value.split())
