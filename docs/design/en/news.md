# News Collection Design

## Status and Scope

This document defines the OSS contract for collecting public company news in
QuantMind. The MVP supports PR Newswire only. It is deliberately small: one
intent-oriented collection operation, one time-window input, deterministic
preprocessing, and explicit partial-failure reporting.

Its public name follows the
[operation naming contract](operations.md): collection returns source-faithful
evidence and remains separate from semantic knowledge extraction.

The primary requirement is that any caller can request a complete, one-shot
collection of a past time window. A daily poll is therefore not a separate
operation; it is simply a short window evaluated on a schedule.

## Design Principles

1. **Express intent, not acquisition mechanics.** Callers ask for news from a
   source and time window. They do not select RSS, listing pages, pagination,
   or article-body rules.
2. **Return honest observations.** QuantMind does not silently deduplicate
   source rows. Repeated source observations remain repeated output records,
   and may share the same stable identity.
3. **Make partial work inspectable.** Item failures are values in the returned
   batch. Invalid inputs still raise before network work starts.
4. **Keep source policy internal.** PR Newswire discovery can change from
   listing pages to another public mechanism without changing the collection
   contract.
5. **Make the supported set explicit.** The MVP uses exhaustive source
   dispatch. It does not expose a provider protocol or a provider registry.
6. **Separate collection from production policy.** Persistence, deduplication,
   rule-based pruning, target schemas, and scheduling belong to the consuming
   data pipeline.

## Public API Contract

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

`NewsWindow` uses timezone-aware timestamps and the half-open interval
`[start, end)`. Both regular collection and historical backfill use this same
call. There are no separate `poll_*`, `backfill_*`, or `fetch_wire_*` public
entry points.

`NewsCollectionCfg` retains the repository's shared `BaseFlowCfg` fields so it
works with the common typed-operation and magic-input tooling. Its only
collection-specific field is `retain_raw_html`, which controls whether fetched
article HTML bytes remain in the result. It defaults to `False`, which is the
conservative and storage-efficient behavior. The article is still fetched,
hashed, parsed, and represented by metadata; only its byte payload is
discarded. The deterministic collector does not otherwise consume the shared
model, tracing, or SDK-run fields.

### Collection records are not knowledge

QuantMind keeps source evidence and semantic extraction as separate contracts:

| Operation | Result | Canonical layer |
|-----------|--------|-----------------|
| `collect_news` | Source-faithful documents, artifacts, failures, and coverage | `quantmind.preprocess` |
| future `extract_news_knowledge` | Extracted financial events | `quantmind.knowledge.News` |

`NewsDocument` is therefore not a `KnowledgeItem`. It carries HTTP evidence,
raw bytes, parsing output, and collection status that remain useful before any
LLM or business schema is chosen. `knowledge.News` is a compact semantic event
with entities, sentiment, materiality, provenance, and an embedding view. The
two may later be composed, but neither substitutes for the other.

## Returned Data

`collect_news` returns a `NewsBatch` with four concepts. The collection
contracts are publicly imported from `quantmind.preprocess`:

- `documents`: successfully collected `NewsDocument` observations;
- `failures`: lightweight `NewsFailure` records for work that could not be
  completed;
- `observed_count`: the number of source rows successfully normalized into
  in-window observations, before article processing;
- `complete`: whether discovery proved coverage of the requested window.

A `NewsDocument` contains the source name, stable identity, canonical URL,
title, publisher, publication time, cleaned Markdown, content hash, ticker
hints, and two evidence artifacts:

- a small discovery artifact representing the public source row;
- an article artifact containing fetch metadata and a content hash. Its
  `bytes` field is `None` unless `retain_raw_html=True`.

`NewsArtifact` is the common evidence shape. It records the content hash,
content type, source and resolved URLs, status, headers, and fetch time, with
an optional byte payload.

Repeated listing rows are not removed. If two rows point to the same release,
the batch contains two observations with the same stable identity. The
consumer can use that identity for an idempotent upsert without losing what
QuantMind actually observed.

## Collection Pipeline

```text
NewsWindow
  -> source dispatch
  -> newest-to-oldest public discovery pages
  -> observations inside [start, end)
  -> linked article fetch
  -> deterministic HTML-to-Markdown normalization
  -> NewsDocument or NewsFailure
  -> NewsBatch
```

PR Newswire discovery is based on its public news-release listing rather than
the latest-items RSS snapshot. Pages are read newest to oldest until an
observation strictly older than the requested start is seen. The strict
boundary matters because several rows with the same minute-level timestamp may
span two pages. This makes a past-day request independently replayable instead
of dependent on a previously saved cursor.

PR Newswire exposes listing timestamps at minute precision. Scheduled callers
should therefore use minute-aligned window bounds, as the live E2E does.

RSS remains a lower-level parser and a live component check. It is not a
high-level `NewsInput`, because feed selection is a source implementation
detail and a bounded feed snapshot cannot prove complete time-window coverage.

The HTTP layer owns bounded retries, backoff, `Retry-After` handling, and
per-host rate limits. PR Newswire-specific URL construction and listing HTML
parsing stay in the PR Newswire source module.

Supported source names form a closed set in `NewsWindow.source`.
`collect_news` dispatches them exhaustively, so widening the input schema
without implementing the corresponding collector fails static verification.

## Failure and Completeness Semantics

Configuration errors raise immediately. Examples include a naive timestamp,
an empty source, an unsupported source, or `end <= start`.

After collection begins, recoverable failures are recorded and independent
items continue. Each `NewsFailure` identifies the source, processing stage,
URL, optional item identity, error category, and message.

`complete=False` when any of the following is true:

- a discovery page could not be fetched or parsed;
- discovery stopped before crossing the window start;

Article failures remain explicit in `failures` but do not change discovery
completeness. A caller can therefore distinguish "the source window was fully
enumerated" from "every observed article was processed." It can safely persist
successful records while separately monitoring and replaying the failed
portion. An empty batch is not considered complete unless discovery has
positively crossed the requested start.

## Responsibility Boundary

QuantMind owns:

- public-source discovery and article acquisition;
- deterministic normalization;
- stable identities and evidence hashes;
- honest batch counts, failure records, and completeness.

The consuming production pipeline owns:

- its schedule and GitHub Action invocation;
- durable storage, watermarks, and idempotent upserts;
- deduplication policy;
- rule-based company-news pruning;
- downstream schemas, enrichment, and database writes;
- shared batch hooks, metrics, and monitoring infrastructure.

This boundary lets a separate ingestion job request the last day in one call,
apply its own rules, and write its own schema without coupling those policies
to this OSS library.

## Verification Contract

Verification has two intentionally separate layers.

### Deterministic verification

`bash scripts/verify.sh` runs deterministic unit tests, linting, typing, and
coverage. News tests use saved HTML/RSS fixtures and mocked HTTP responses.
They cover window validation, time boundaries, pagination, duplicate
observations, raw-byte retention, retries, partial failures, and completeness.

### Live news E2E

`python scripts/verify_news_e2e.py` is the component-specific live-network news
smoke test. It performs two bounded checks:

1. fetch and parse the official PR Newswire RSS feed;
2. discover PR Newswire listing observations for the preceding 24 hours and
   prove that discovery crossed the window start.

The E2E check never fetches article pages. It prints component-level PASS/FAIL
records and a compact observation/page/failure summary, then exits non-zero if
RSS is empty or invalid, listing discovery is empty or incomplete, or a
component raises. This keeps the check lightweight while detecting public
source or parser drift.

The `news` job in `.github/workflows/e2e.yml` runs this check once daily, on
manual dispatch, and on pull requests that change its precise dependency paths.
The required `.github/workflows/ci.yml` workflow remains network-free so local
development and unit tests stay deterministic.

## Non-Goals and Extension Rule

The MVP does not include GlobeNewswire, Business Wire, authenticated feeds,
continuous cursors, storage, deduplication, business materiality scoring, or a
generic batch-operation base class.

When a second source is implemented, compare its real behavior with PR
Newswire first. Extract a shared provider interface only for behavior the two
implementations genuinely share. The public
`collect_news(NewsWindow, *, cfg)` contract should remain unchanged.

## Adding a Public News Source

A coding agent adding a second source follows this closed checklist:

1. Add the source name to `NewsWindow.source`.
2. Add one private `quantmind/preprocess/<source>.py` collector.
3. Add one explicit branch to `collect_news`; the exhaustive type check must
   remain green.
4. Add fixture-based success, boundary, duplicate-observation,
   partial-failure, and completeness tests for the source.
5. Add a routing test proving only the selected collector is called.
6. Update the supported-source table in `docs/README.md`, this design, and the
   focused example if its common path changes.
7. Add or extend a component-specific live verifier and its named job in the
   existing `.github/workflows/e2e.yml` when the integration depends on a
   public network endpoint. Add its command to `docs/README.md`; do not add the
   command to root agent guidance.

Only after two real collectors expose shared behavior should a common
`Protocol` be considered. A new source must never be added only to the input
Literal: static verification is expected to reject that incomplete change.
