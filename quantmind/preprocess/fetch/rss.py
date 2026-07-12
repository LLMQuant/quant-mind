"""RSS/Atom feed fetch and parsing helpers for news ingestion.

This module deliberately stays in the fetch layer: it downloads feed XML and
turns it into typed feed-item metadata. Article-body extraction and source
normalisation live in ``quantmind.preprocess.news``.
"""

from dataclasses import dataclass, field
from datetime import datetime
from xml.etree import ElementTree

from quantmind.preprocess.fetch._types import Fetched
from quantmind.preprocess.fetch.http import (
    DEFAULT_USER_AGENT,
    HttpFetcher,
    fetch_url,
)
from quantmind.preprocess.time import parse_news_datetime


@dataclass(frozen=True, slots=True)
class FeedItem:
    """One RSS/Atom entry discovered in a feed."""

    title: str
    url: str | None = None
    id: str | None = None
    published_at: datetime | None = None
    summary_html: str | None = None
    content_html: str | None = None
    source_feed_url: str | None = None
    raw: dict[str, str] = field(default_factory=dict)
    raw_xml: bytes = b""


@dataclass(frozen=True, slots=True)
class RawFeed:
    """Parsed RSS/Atom feed metadata and items."""

    feed_url: str | None
    title: str | None = None
    items: tuple[FeedItem, ...] = ()
    content_type: str = "application/xml"
    headers: dict[str, str] = field(default_factory=dict)
    fetched: Fetched | None = None


async def fetch_rss_feed(
    url: str,
    *,
    timeout: float = 30.0,
    max_bytes: int = 5_000_000,
    user_agent: str = DEFAULT_USER_AGENT,
    fetcher: HttpFetcher | None = None,
) -> RawFeed:
    """Fetch an RSS/Atom feed and parse its entries.

    Args:
        url: Absolute RSS/Atom feed URL.
        timeout: Per-request timeout in seconds.
        max_bytes: Hard ceiling on response body size.
        user_agent: Optional User-Agent override.
        fetcher: Optional shared HTTP fetcher and host-rate state.

    Returns:
        Parsed feed metadata and entries.

    Raises:
        ValueError: If the feed XML is unsupported or malformed.
        httpx.HTTPError: For network / status / timeout failures.
    """
    if fetcher is None:
        raw = await fetch_url(
            url,
            timeout=timeout,
            max_bytes=max_bytes,
            user_agent=user_agent,
        )
    else:
        raw = await fetcher.fetch_url(
            url,
            timeout=timeout,
            max_bytes=max_bytes,
            user_agent=user_agent,
        )
    return parse_feed(
        raw.bytes,
        feed_url=url,
        content_type=raw.content_type,
        headers=raw.headers,
        fetched=raw,
    )


def parse_feed(
    xml_bytes: bytes,
    *,
    feed_url: str | None = None,
    content_type: str = "application/xml",
    headers: dict[str, str] | None = None,
    fetched: Fetched | None = None,
) -> RawFeed:
    """Parse RSS 2.x or Atom XML bytes into a ``RawFeed``.

    Args:
        xml_bytes: Raw XML payload.
        feed_url: Origin feed URL, when known.
        content_type: MIME type observed for the feed.
        headers: Selected HTTP response headers.
        fetched: Original fetched response metadata, when available.

    Returns:
        Parsed feed metadata and items.

    Raises:
        ValueError: If the XML is empty, malformed, or not RSS/Atom.
    """
    if not xml_bytes.strip():
        raise ValueError("empty feed payload")
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        raise ValueError(f"malformed feed XML: {exc}") from exc

    root_name = _local_name(root.tag)
    if root_name == "rss":
        title, items = _parse_rss(root, feed_url)
    elif root_name == "feed":
        title, items = _parse_atom(root, feed_url)
    else:
        raise ValueError(f"unsupported feed root: {root_name!r}")

    return RawFeed(
        feed_url=feed_url,
        title=title,
        items=tuple(items),
        content_type=content_type,
        headers=dict(headers or {}),
        fetched=fetched,
    )


def _parse_rss(
    root: ElementTree.Element,
    feed_url: str | None,
) -> tuple[str | None, list[FeedItem]]:
    channel = _first_child(root, "channel") or root
    title = _child_text(channel, "title")
    items: list[FeedItem] = []
    for item in _children(channel, "item"):
        item_title = _child_text(item, "title") or ""
        url = _child_text(item, "link")
        item_id = _child_text(item, "guid") or url
        published_at = _parse_datetime_or_none(
            _child_text(item, "pubDate")
            or _child_text(item, "date")
            or _child_text(item, "published")
        )
        summary = _child_text(item, "description")
        content = _child_text(item, "encoded") or _child_text(item, "content")
        items.append(
            FeedItem(
                title=item_title,
                url=url,
                id=item_id,
                published_at=published_at,
                summary_html=summary,
                content_html=content,
                source_feed_url=feed_url,
                raw={
                    k: v
                    for k, v in {
                        "guid": item_id,
                        "pubDate": _child_text(item, "pubDate"),
                    }.items()
                    if v
                },
                raw_xml=ElementTree.tostring(item, encoding="utf-8"),
            )
        )
    return title, items


def _parse_atom(
    root: ElementTree.Element,
    feed_url: str | None,
) -> tuple[str | None, list[FeedItem]]:
    title = _child_text(root, "title")
    items: list[FeedItem] = []
    for entry in _children(root, "entry"):
        item_title = _child_text(entry, "title") or ""
        url = _atom_link(entry)
        item_id = _child_text(entry, "id") or url
        published_at = _parse_datetime_or_none(
            _child_text(entry, "published") or _child_text(entry, "updated")
        )
        summary = _child_text(entry, "summary")
        content = _child_text(entry, "content")
        items.append(
            FeedItem(
                title=item_title,
                url=url,
                id=item_id,
                published_at=published_at,
                summary_html=summary,
                content_html=content,
                source_feed_url=feed_url,
                raw={
                    k: v
                    for k, v in {
                        "id": item_id,
                        "updated": _child_text(entry, "updated"),
                    }.items()
                    if v
                },
                raw_xml=ElementTree.tostring(entry, encoding="utf-8"),
            )
        )
    return title, items


def _atom_link(entry: ElementTree.Element) -> str | None:
    fallback: str | None = None
    for child in _children(entry, "link"):
        href = child.attrib.get("href")
        if not href:
            continue
        rel = child.attrib.get("rel", "alternate").strip().lower()
        if rel == "alternate":
            return href.strip()
        fallback = fallback or href.strip()
    return fallback


def _parse_datetime_or_none(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parse_news_datetime(value)
    except ValueError:
        return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(
    element: ElementTree.Element,
    name: str,
) -> list[ElementTree.Element]:
    return [child for child in list(element) if _local_name(child.tag) == name]


def _first_child(
    element: ElementTree.Element,
    name: str,
) -> ElementTree.Element | None:
    for child in list(element):
        if _local_name(child.tag) == name:
            return child
    return None


def _child_text(element: ElementTree.Element, name: str) -> str | None:
    child = _first_child(element, name)
    if child is None:
        return None
    text = "".join(child.itertext()).strip()
    return text or None
