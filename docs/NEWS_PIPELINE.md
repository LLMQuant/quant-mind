# News Pipeline

QuantMind provides deterministic building blocks for fetching, parsing, and
normalizing public financial news. The current wire-provider MVP supports PR
Newswire. The provider contract is extensible, but other wire services are not
built-in or verified yet.

This guide describes the public OSS behavior implemented today. It does not
assume private feeds, credentials, persistence infrastructure, or internal
services.

## What the Pipeline Can Do

| Stage | Public API | Result |
| --- | --- | --- |
| HTTP acquisition | `fetch_url`, `HttpFetcher`, `FetchPolicy` | Bounded response bytes and fetch metadata |
| Feed parsing | `fetch_rss_feed`, `parse_feed` | Typed RSS 2.x or Atom items |
| Article acquisition | `feed_item_to_news_document` | Explicit feed-body or linked-article text |
| Normalization | `preprocess_news_document`, `preprocess_feed_item` | Deterministic `NewsCandidate` |
| Wire ingestion | `fetch_wire_documents` | PR Newswire documents plus non-fatal failures |

The main wire path is:

```text
Public RSS/Atom feed
  -> shared HTTP fetcher
  -> typed feed items
  -> provider mapping
  -> linked article fetch
  -> deterministic news normalization
  -> WireDocument + optional WireFetchFailure records
```

### HTTP robustness

`FetchPolicy` can opt a caller into:

- bounded retries for transport errors, HTTP 408, HTTP 429, and HTTP 5xx;
- exponential backoff with configurable jitter;
- `Retry-After` handling for HTTP 429 and HTTP 503;
- per-host concurrency limits and minimum request spacing.

Calling `fetch_url` without a policy preserves its one-shot behavior. Reuse one
`HttpFetcher` when several requests should share a connection pool and the same
per-host rate state.

### RSS and Atom parsing

`fetch_rss_feed` and `parse_feed` support RSS 2.x and Atom. Each `FeedItem`
retains normalized fields, selected source metadata, and the raw XML entry.

### Explicit body acquisition

News feed items use an explicit body-source decision:

- `body_source="feed"` trusts the content included in the feed and performs no
  article request;
- `body_source="article"` always follows the item URL, even when the feed
  contains a non-empty teaser.

The default is `feed`. The built-in PR Newswire adapter uses `article` because
its public feed descriptions may be teasers.

### Deterministic normalization

`NewsCandidate` normalization currently includes:

- Unicode, whitespace, and duplicate-line cleanup;
- canonical source URLs with common tracking parameters removed;
- UTC publication timestamps;
- stable content hashes and source deduplication keys;
- exchange-qualified ticker hints such as `NASDAQ: NVDA` or `NYSE: IBM`.

Ticker extraction only produces hints. Instrument resolution and validation
remain downstream responsibilities.

### Replayable wire documents

Each successful `WireDocument` contains:

- the raw feed entry;
- raw article HTML when an article request was used;
- source and fetch metadata;
- cleaned Markdown;
- provider identity, payload ID, canonical URL, title, publisher, publication
  time, content hash, stable identity, and ticker hints.

Individual feed or article failures do not fail the whole call. They are
returned as lightweight `WireFetchFailure` values alongside successful
documents.

## Examples

### Normalize an existing news document

This path performs no network request and is useful when the caller already
has the article text.

```python
from datetime import datetime, timezone

from quantmind.preprocess import RawNewsDocument, preprocess_news_document

raw = RawNewsDocument(
    title="Example Corp Reports Results",
    body_text=(
        "Example Corp (NASDAQ: EXMPL) reported fictional quarterly revenue."
    ),
    source_url="https://example.com/releases/results?utm_source=rss",
    publisher="Example Publisher",
    published_at=datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc),
    payload_id="release-123",
)

candidate = preprocess_news_document(raw)

print(candidate.source_url)
print(candidate.dedup_key)
print([hint.symbol for hint in candidate.ticker_hints])
```

A runnable version lives at
[`examples/preprocess/01_news_pr_wire.py`](../examples/preprocess/01_news_pr_wire.py).

### Fetch and normalize a custom RSS or Atom feed

Choose the body source based on the feed contract. Use `article` when entries
only contain teasers.

```python
import asyncio

from quantmind.preprocess import (
    FetchPolicy,
    HttpFetcher,
    fetch_rss_feed,
    preprocess_feed_item,
)


async def main() -> None:
    policy = FetchPolicy(
        max_attempts=3,
        max_concurrency_per_host=2,
        min_interval_seconds=0.25,
    )
    async with HttpFetcher(policy=policy) as fetcher:
        feed = await fetch_rss_feed(
            "https://example.com/news.xml",
            fetcher=fetcher,
        )
        candidates = [
            await preprocess_feed_item(
                item,
                publisher="Example Publisher",
                body_source="article",
                fetcher=fetcher,
            )
            for item in feed.items
        ]

    print(f"candidates={len(candidates)}")


asyncio.run(main())
```

### Fetch the public PR Newswire feed

`fetch_wire_documents` uses one shared fetcher for the feed and article
requests. The result contains successful documents and recorded failures.

```python
import asyncio

from quantmind.preprocess import (
    PR_NEWSWIRE,
    FetchPolicy,
    WireFeedConfig,
    fetch_wire_documents,
)


async def main() -> None:
    result = await fetch_wire_documents(
        WireFeedConfig(
            provider=PR_NEWSWIRE,
            feed_urls=(
                "https://www.prnewswire.com/rss/news-releases-list.rss",
            ),
            fetch_policy=FetchPolicy(min_interval_seconds=0.25),
        )
    )

    print(f"documents={result.success_count}")
    print(f"failures={result.failure_count}")

    if result.documents:
        document = result.documents[0]
        print(document.identity)
        print(document.cleaned_markdown[:200])
        print(document.raw_feed_entry.content_hash)
        if document.raw_article is not None:
            print(document.raw_article.content_hash)

    for failure in result.failures:
        print(failure.stage, failure.error_type, failure.source_url)


asyncio.run(main())
```

A focused runnable version lives at
[`examples/preprocess/02_wire_ingestion.py`](../examples/preprocess/02_wire_ingestion.py).

## Verification

The normal test suite is deterministic and does not require public network
access:

```bash
python -m pytest --no-cov tests/preprocess
bash scripts/verify.sh
```

The live smoke test is separate and performs real PR Newswire feed and article
requests:

```bash
python scripts/smoke_wire.py
```

The smoke checklist verifies that the current public feed produces documents,
raw feed entries, raw articles, cleaned Markdown, unique identities, and no
recorded failures.

## Current Non-Goals

The current pipeline does not provide:

- built-in GlobeNewswire or Business Wire adapters;
- persistent cursors, watermarks, or time-window pagination;
- scheduling or durable storage;
- a shared batch-operation base class, hooks, monitoring, or metrics;
- authenticated or private provider endpoints;
- automatic company or instrument resolution;
- downstream news-card generation.
