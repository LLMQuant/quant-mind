# Build a source-first paper result

## Quick Summary

- **Purpose**: Define how one exact PDF revision becomes a durable page-aware chunk set and a cited global summary.
- **Read when**: Changing `paper_flow`, paper inputs, summarization limits, citation validation, or the end-to-end paper verifier.
- **Status**: Implemented by `quantmind.flows.paper_flow` for PDF-backed arXiv, HTTP, and local inputs.
- **Core rule**: Preserve and validate the source revision and chunk set before any model-generated summary is accepted.
- **Canonical models**: [Paper source and artifact design](../knowledge/paper.md).

## Contents

- [Contract](#contract)
- [Execution Order](#execution-order)
- [Source Revision](#source-revision)
- [Chunk Set](#chunk-set)
- [Cited Global Summary](#cited-global-summary)
- [Bounded Model Calls](#bounded-model-calls)
- [Failure Semantics](#failure-semantics)
- [Persistence and Retrieval](#persistence-and-retrieval)
- [Verification Slice](#verification-slice)
- [Out of Scope](#out-of-scope)

## Contract

`paper_flow(input, *, cfg)` returns one validated `PaperFlowResult`:

```text
PaperFlowResult
├── source_revision: PaperSourceRevision
├── chunk_set: PaperChunkSet
└── global_summary: PaperGlobalSummary
```

The result is source-first. It does not return a model-authored paper tree, and V1 does not define `PaperTree`. The source revision is an immutable anchor for independently versioned artifacts. A different splitter configuration may produce another chunk set for the same source; a different model, prompt, input chunk set, or output bound may produce another summary. These versions coexist instead of overwriting each other.

V1 accepts:

| Input | Behavior |
|---|---|
| `ArxivIdentifier` | Resolve and fetch the exact PDF revision. The canonical source records a versioned arXiv ID such as `1706.03762v7`. |
| `HttpUrl` | Fetch the URL and require PDF content. |
| `LocalFilePath` | Read the local file and require PDF content. |
| `RawText` | Reject because it has no physical page evidence. |
| `DoiIdentifier` | Reject until an exact open-PDF resolver exists. |

## Execution Order

The operation has a strict order:

1. Resolve the input to exact bytes and source metadata.
2. Parse the PDF into ordered physical pages, blocks, screenshots, and extracted images.
3. Build and validate `PaperSourceRevision`, including content-addressed asset references and blobs.
4. Chunk each page with LlamaIndex while retaining page and character spans.
5. Build and validate `PaperChunkSet` with code-owned IDs and membership.
6. Give a coordinator agent the bounded chunk manifest and a research-agent tool.
7. Require the coordinator to delegate complete chunk coverage to one or more bounded research subagents, with independent ranges eligible for parallel execution.
8. Let the coordinator synthesize typed worker reports and make bounded overlapping follow-up calls when needed.
9. Resolve model-returned chunk/page coordinates into canonical citations in code.
10. Build and validate `PaperGlobalSummary` and the cross-artifact `PaperFlowResult`.

A summarization failure occurs after source and chunks exist in memory, but `paper_flow` returns no partial success value. Persistence is a separate explicit operation.

## Source Revision

`PaperSourceRevision` owns the exact fetched bytes and deterministic parser result. Its ID is derived from the source SHA-256 hash. It records:

- typed source metadata, financial `as_of`, `available_at`, publication time, exact arXiv revision, title, and authors;
- parser name, parser version, cleanup version, and every 1-based physical page;
- page text, source blocks, bounding boxes, screenshot references, and extracted-image references;
- content-addressed references for the raw PDF and every retained visual asset;
- exact asset bytes while crossing from the flow to persistence.

The raw asset hash must equal the source hash. Every asset ID is derived from source revision, asset kind, page, and content hash. Page references must resolve to assets on that page. Canonical JSON excludes blob bytes; [`LocalKnowledgeLibrary.put_paper()`](../library/local.md) stores them in linked SQLite rows.

For arXiv, the source must record an exact version suffix. Fetch time is never used as publication time. `available_at` uses exact revision metadata when supplied and otherwise falls back to the earliest reliable observation supported by the input.

## Chunk Set

`PaperChunkSet` is built before summarization. Its producer identity contains the LlamaIndex sentence-splitter version, chunk size, and overlap. The producer configuration hash and source revision deterministically identify the artifact.

Each `PaperChunk` has a stable code-owned ID, contiguous position, exact text hash, and one or more `PaperSourceSpan` values. A span retains its 1-based page number, page-local character offsets, block boxes, and visual asset IDs. The chunk-set content hash covers ordered membership, content hashes, and source spans.

Chunk IDs, artifact IDs, hashes, positions, and membership are not model outputs. Repeating the same source and configuration produces the same source, chunk-set, and chunk IDs.

## Cited Global Summary

The model returns only a draft containing summary prose and citation coordinates: chunk index, page number, and an optional quote. It never chooses canonical UUIDs, artifact lineage, source facts, or timestamps.

Code accepts the draft only when:

- every chunk index exists;
- every cited page is present in that chunk's source spans;
- every supplied quote occurs verbatim in the cited chunk;
- citation count meets `summary_min_citations`;
- distinct cited-page count meets `summary_min_pages`.

Code then creates `PaperCitation` values and a `PaperGlobalSummary`. Its producer identity includes model, prompt version, input chunk-set ID, instructions hash, and maximum output tokens. Its lineage contains the exact input chunk-set locator. Summary citations must resolve through that chunk set to source pages.

## Bounded Model Calls

`PaperFlowCfg` makes coordinator and nested-research bounds explicit:

- `max_summary_tool_calls` caps research-subagent invocations, including focused follow-ups;
- `max_summary_concurrency` bounds simultaneous research-subagent runs;
- `max_summary_worker_turns` caps each nested agent run;
- `max_summary_worker_output_tokens` caps each structured worker report;
- `max_summary_input_tokens` caps the manifest, worker chunk inputs, and worker reports returned to the coordinator;
- `max_summary_output_tokens` caps the coordinator's final structured response;
- `max_summary_total_output_tokens` caps worker reports and final output together;
- `max_turns` caps coordinator turns and defaults to 16;
- `timeout_seconds` bounds the complete summarization operation;
- `summary_prompt_version` and `summary_instructions` version the semantic producer.

The coordinator uses the Agents SDK manager-style pattern: `paper_chunk_researcher` is exposed through `Agent.as_tool()`, so the coordinator retains responsibility for the final summary while each specialist receives only its requested range. The manifest provides code-generated groups of at most eight consecutive chunks. Independent calls may run in parallel, and the coordinator may request bounded overlapping follow-up research. Code rejects missing full-chunk coverage, out-of-scope worker citations, invalid pages or quotes, and any call, concurrency, token, or runtime excess.

## Failure Semantics

- Non-PDF input raises `UnsupportedContentTypeError`.
- Unresolved DOI input raises `NotImplementedError`.
- Fetching, parsing, or missing parser assets raise their source error and produce no result.
- Empty chunk output is invalid.
- Invalid or insufficient summary citations raise `PaperCitationValidationError`.
- Missing worker coverage, invalid worker evidence, and call, token, output, concurrency, or timeout violations fail the summary operation.
- Any canonical identity, content hash, membership, lineage, or cross-artifact mismatch fails Pydantic validation.

No failure is converted into a partially valid `PaperFlowResult`. Callers may retry with the same source and producer settings; stable IDs make successful repeated runs idempotent.

## Persistence and Retrieval

The flow performs no library write. Callers explicitly pass its result to `LocalKnowledgeLibrary.put_paper()`. That method prepares all required summary and chunk embeddings before one SQLite transaction, so provider failure cannot leave a partial paper.

Collection search uses library-owned projections. A summary hit resolves to `PaperGlobalSummary`; a chunk hit resolves to `PaperChunk`. `SemanticHit.locator` carries source revision, artifact, artifact kind, and optional member ID. `SemanticHit.projection` explains the rebuildable embedding projection used for ranking.

## Verification Slice

Offline tests use fixed generated PDFs and fake summary/embedding providers. They verify exact source bytes, page evidence, stable IDs, citation rejection, bounds, atomic writes, reopen behavior, projection reuse, multi-version coexistence, search, and locator resolution.

The bounded live slice is:

```bash
python scripts/verify_pdf_rag_e2e.py
```

It fetches exact arXiv revision `1706.03762v7`, parses at least 15 physical pages, creates multiple page-aware chunks, delegates complete coverage to bounded `gpt-4o-mini` research subagents, synthesizes a cited global summary, persists with `text-embedding-3-small`, closes and reopens the database, runs summary and chunk searches, resolves every returned locator, and prints useful summary, page, citation, and score diagnostics. The dedicated `paper-flow` job in `.github/workflows/e2e.yml` owns this non-required public-network check.

## Out of Scope

- model-authored canonical IDs, hashes, lineage, or source metadata;
- a V1 paper tree or PageIndex structure;
- HTML, Markdown, raw-text, or scanned-document OCR paper ingestion;
- DOI-to-open-PDF resolution;
- question answering over search results;
- hidden or unbounded model calls;
- implicit persistence from `paper_flow`.
