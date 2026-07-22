# News Collection Design

## Quick Summary

- **Purpose**: Define how QuantMind collects public news and records proof of what each source returned.
- **Read when**: Changing `collect_news`, `NewsWindow`, a news source, the `complete` flag, or news checks.
- **Status**: Current design; the open-source package currently supports PR Newswire.
- **Core rule**: Collection returns documents, failures, and whether the full time window was covered. Other code turns documents into knowledge, stores them, schedules runs, and removes irrelevant news.

## Contents

- [Scope and Current Support](#scope-and-current-support)
- [Design Principles](#design-principles)
- [Public API](#public-api)
- [Returned Data](#returned-data)
- [Collection Steps](#collection-steps)
- [Failures and the `complete` Flag](#failures-and-the-complete-flag)
- [Who Owns What](#who-owns-what)
- [How to Verify](#how-to-verify)
- [Out of Scope and When to Add Shared Code](#out-of-scope-and-when-to-add-shared-code)
- [Adding a Public News Source](#adding-a-public-news-source)

## Scope and Current Support

This page defines how the open-source package collects public company news. The
first version supports PR Newswire only. It has one collection function, one
time-window input, repeatable HTML cleanup, and visible item failures.

Its public name follows the
[operation naming rules](../operations/naming.md): collection returns the news
as published plus fetch details. A separate operation turns those documents
into structured knowledge.

The primary requirement is that any caller can request a complete, one-shot
collection of a past time window. A daily poll is therefore not a separate
operation; it is simply a short window evaluated on a schedule.

## Design Principles

1. **Ask for news, not fetch details.** Callers choose a source and time window.
   They do not choose RSS, listing pages, how to move between pages, or article
   parsing rules.
2. **Return every source row.** QuantMind does not silently remove duplicate
   source rows. Repeated rows remain repeated output records,
   and may share the same stable ID.
3. **Show partial failures.** The returned batch includes item failures. Invalid
   inputs still raise before network work starts.
4. **Hide source implementation details.** PR Newswire may later use a public
   mechanism other than listing pages without changing the public function.
5. **List supported sources explicitly.** The first version selects from a
   closed source list. It does not expose a provider plugin API or registry.
6. **Keep downstream choices separate.** The calling pipeline owns storage,
   duplicate handling, relevance rules, output formats, and scheduling.

## Public API

Callers use one entry point:

```python
from datetime import datetime, timezone

from quantmind.configs import NewsCollectionCfg, NewsWindow
from quantmind.flows import collect_news

batch = await collect_news(
    NewsWindow(
        source="pr-newswire",
        start=datetime(2026, 7, 13, tzinfo=timezone.utc),
        end=datetime(2026, 7, 14, tzinfo=timezone.utc),
    ),
    cfg=NewsCollectionCfg(retain_raw_html=False),
)
```

`NewsWindow` requires timezone-aware timestamps. Its `[start, end)` interval
includes `start` and excludes `end`. Regular collection and historical runs
use the same call; there are no separate `poll_*`, `backfill_*`, or
`fetch_wire_*` public functions.

`NewsCollectionCfg` keeps the shared `BaseFlowCfg` fields so it works with the
repository's shared config handling. Its only
collection-specific field is `retain_raw_html`, which controls whether fetched
article HTML bytes remain in the result. It defaults to `False`, which is the
storage-efficient behavior. The article is still fetched, hashed, parsed, and
described by fetch details; only its byte payload is discarded. The collector
does not otherwise use the shared model, tracing, or SDK-run fields.

### Collection records are not knowledge

QuantMind keeps collected documents separate from structured knowledge:

| Operation | Result | Owning package |
|-----------|--------|-----------------|
| `collect_news` | Source documents, fetch details, failures, and whether the full window was scanned | `quantmind.preprocess` |
| future `extract_news_knowledge` | Extracted financial events | `quantmind.knowledge.News` |

`NewsDocument` is therefore not a `KnowledgeItem`. It carries HTTP details,
raw bytes, parsing output, and collection status that remain useful before any
LLM or business data format is chosen. `knowledge.News` is a structured
financial event with entities, sentiment, financial importance
(`materiality`), source links, and text for embeddings. A pipeline may use both
types, but one cannot replace the other.

## Returned Data

`collect_news` returns a `NewsBatch` with four fields. Import these public types
from `quantmind.preprocess`:

- `documents`: successfully collected `NewsDocument` observations;
- `failures`: lightweight `NewsFailure` records for work that could not be
  completed;
- `observed_count`: the number of source rows successfully converted into
  in-window observations, before article processing;
- `complete`: whether the listing scan covered the full requested window.

A `NewsDocument` contains the source name, stable ID, preferred article URL,
title, publisher, publication time, cleaned Markdown, content hash, ticker
hints, and two `NewsArtifact` records:

- a small record of the public listing row;
- an article record containing fetch details and a content hash. Its
  `bytes` field is `None` unless `retain_raw_html=True`.

`NewsArtifact` records what was fetched. It contains the content hash,
content type, source and resolved URLs, status, headers, and fetch time, with
an optional byte payload.

Repeated listing rows are not removed. If two rows point to the same release,
the batch contains two observations with the same stable ID. A consumer can
use that ID to update the same database row safely while still recording what
QuantMind observed.

## Collection Steps

```text
NewsWindow
  -> select the source collector
  -> scan public listing pages from newest to oldest
  -> keep rows inside [start, end)
  -> fetch each linked article
  -> convert HTML to Markdown
  -> NewsDocument or NewsFailure
  -> NewsBatch
```

PR Newswire discovery is based on its public news-release listing rather than
the latest-items RSS snapshot. Pages are read newest to oldest until an
observation strictly older than the requested start is seen. The strict
boundary matters because several rows with the same minute-level timestamp may
span two pages. A caller can therefore rerun a past-day request without a
previously saved cursor.

PR Newswire exposes listing timestamps at minute precision. Scheduled callers
should therefore use minute-aligned window bounds, as the live end-to-end check
does.

RSS remains an internal parser and a live check. It is not a public
`NewsInput`: choosing a feed is an implementation detail, and a limited feed
snapshot cannot prove that it covered a complete time window.

The HTTP layer limits retries, waits between attempts, honors `Retry-After`,
and limits requests per host. PR Newswire-specific URL construction and
listing HTML parsing stay in the PR Newswire source module.

`NewsWindow.source` lists every supported source. `collect_news` has an
explicit branch for each one, so the type checker catches a source name added
without a matching collector.

## Failures and the `complete` Flag

Configuration errors raise immediately. Examples include a naive timestamp,
an empty source, an unsupported source, or `end <= start`.

After collection begins, one item failure does not stop independent items.
Each `NewsFailure` records the source, failed step, URL, optional item ID, error
category, and message.

`complete=False` when any of the following is true:

- a listing page could not be fetched or parsed;
- the listing scan stopped before crossing the window start.

Article failures stay in `failures` but do not change whether all listing rows
were found. A caller can distinguish "the full time window was scanned" from
"every found article was processed." It can store successful records and retry
failed articles separately. An empty batch is complete only when the listing
scan crossed the requested start.

## Who Owns What

QuantMind owns:

- scanning public listings and fetching articles;
- repeatable HTML cleanup;
- stable IDs and content hashes;
- honest batch counts, failure records, and completeness.

The consuming production pipeline owns:

- its schedule and when its GitHub Action runs;
- database storage, saved progress, and safe repeated writes;
- rules for removing duplicates;
- rules that remove irrelevant company news;
- later data formats, added fields, and database writes;
- shared batch callbacks, metrics, and monitoring.

This split lets a separate ingestion job request the last day in one call,
apply its own rules, and write its own data format without adding those choices
to the open-source library.

## How to Verify

Use separate offline and live checks.

### Offline repository checks

`bash scripts/verify.sh` runs repeatable unit tests, linting, typing, and
coverage. News tests use saved HTML/RSS files and mocked HTTP responses.
They cover window validation, time edges, multi-page listings, duplicate
observations, raw-byte retention, retries, partial failures, and completeness.

### Live end-to-end news check

`python scripts/verify_news_e2e.py` is the news live-network check. It performs
three limited checks:

1. fetch and parse the official PR Newswire RSS feed;
2. scan PR Newswire listing rows for the preceding 24 hours and prove that the
   scan crossed the window start;
3. fetch up to 25 unique articles and find the first supported exchange-coded
   symbol in every group that has one (100% recall).

Ticker-hint extraction supports one exchange-qualified symbol and parenthesized
comma-separated symbol lists that share a supported exchange prefix. A shared
group ends at a semicolon, conjunction, or closing parenthesis. Additional list
members record a reconstructed `EXCHANGE: SYMBOL` raw value; unsupported
exchanges remain outside the whitelist policy.

The ticker check uses a separate comparison regex rather than the production
extractor. It accepts plain, Markdown-link, and emphasis-wrapped symbols. It
recognizes comma-separated symbols that share a supported exchange prefix and
ignores exchanges that the extractor does not support. The script prints
PASS/FAIL plus short listing, article,
failure, and recall summaries. It fails when RSS or listing data is invalid,
no sampled article parses, or supported symbols fall below 100% recall. A
parsed sample with no supported symbols reports `SKIP` and still passes.

The `news` job in `.github/workflows/e2e.yml` runs this check once daily, when
started manually, and on pull requests that change its direct dependencies.
The required `.github/workflows/ci.yml` workflow remains network-free so local
development and unit tests remain repeatable.

## Out of Scope and When to Add Shared Code

The first version does not include GlobeNewswire, Business Wire, authenticated
feeds, continuous cursors, storage, duplicate removal, financial importance
scoring, or a generic batch-operation base class.

When a second source is implemented, compare its real behavior with PR
Newswire first. Add a shared collector interface only for behavior that both
working collectors actually share. Keep the public
`collect_news(NewsWindow, *, cfg)` call unchanged.

## Adding a Public News Source

A coding agent adding a second source follows this checklist:

1. Add the source name to `NewsWindow.source`.
2. Add one private `quantmind/preprocess/<source>.py` collector.
3. Add one explicit branch to `collect_news`; the type checker must pass.
4. Add saved-input tests for success, time edges, duplicate observations,
   partial failures, and the `complete` flag.
5. Add a test proving that only the selected collector is called.
6. Update the supported-source table in `docs/README.md`, this design, and the
   focused example if its common path changes.
7. Add or extend a component-specific live verifier and its named job in the
   existing `.github/workflows/e2e.yml` when the integration depends on a
   public network endpoint. Add its command to `docs/README.md`; do not add the
   command to root agent guidance.

Only after two real collectors share behavior should they use a common Python
`Protocol`. Never add a source only to the input `Literal`; the type checker is
expected to reject that incomplete change.
