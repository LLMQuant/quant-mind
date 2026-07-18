# Model paper sources and artifacts independently

## Quick Summary

- **Purpose**: Define canonical Paper Flow V1 source, chunk, summary, citation, and locator models.
- **Read when**: Changing `quantmind.knowledge.paper`, stable paper identities, artifact lineage, or paper search resolution.
- **Status**: Implemented by `quantmind.knowledge.paper` and persisted by `quantmind.library`.
- **Core rule**: Source revisions are immutable anchors; derived artifacts are independently versioned and never own retrieval vectors.

## Contents

- [Model Layers](#model-layers)
- [Stable Identity](#stable-identity)
- [Source and Asset Integrity](#source-and-asset-integrity)
- [Artifact Versioning](#artifact-versioning)
- [Citation and Lineage Integrity](#citation-and-lineage-integrity)
- [Retrieval Boundary](#retrieval-boundary)
- [Compatibility Boundary](#compatibility-boundary)

## Model Layers

Paper Flow V1 separates three layers:

| Layer | Canonical aggregate | Addressable members | Purpose |
|---|---|---|---|
| Exact source | `PaperSourceRevision` | `PaperAssetRef`, `PaperParsedPage`, `PaperParsedBlock` | Preserve fetched bytes, page-aware parser output, metadata, and visual evidence. |
| Deterministic artifact | `PaperChunkSet` | `PaperChunk`, `PaperSourceSpan` | Record one exact chunking of the source before any summary call. |
| Semantic artifact | `PaperGlobalSummary` | `PaperCitation` | Store one independently versioned model summary with resolvable chunk/page evidence. |

`PaperFlowResult` validates one compatible source, chunk set, and summary combination. It is a transfer result, not a fourth stored artifact.

All models are frozen Pydantic values with `extra="forbid"`. Canonical values contain no embedding vectors, provider node objects, or storage handles.

## Stable Identity

IDs are generated and checked in code:

- source revision ID: UUIDv5 over the exact source SHA-256 hash;
- asset ID: UUIDv5 over source revision, asset kind, page, and asset content hash;
- artifact ID: UUIDv5 over source revision, artifact kind, and producer configuration hash;
- chunk ID: UUIDv5 over chunk-set ID, position, content hash, and source-span hash.

Producer configuration is canonical JSON with sorted keys before SHA-256 hashing. Chunk-set content hashes cover ordered chunk membership and spans. Summary content hashes cover summary prose and ordered citations.

These identities make an identical run idempotent. They also keep a changed splitter or summary producer from overwriting an older artifact.

## Source and Asset Integrity

`PaperSourceRevision` requires a typed `SourceRef` whose `content_hash` equals the parsed manifest hash and raw-asset hash. arXiv sources require an exact revision suffix. Pages are contiguous and 1-based. Every page visual reference must name a known asset from that page.

`PaperAssetRef` records media type, content hash, byte length, kind, and optional page. Exact blobs are keyed by content hash while the result crosses into persistence. When blobs are loaded, every reference must have bytes with matching length and SHA-256 hash.

The source's canonical JSON excludes blobs. This keeps canonical hashes stable and reviewable while allowing SQLite to store exact bytes in a normalized linked table. Rehydration checks both directions: stored blob bytes must match their table hashes, and table asset metadata must match the canonical source manifest.

Every chunk span is also checked against that manifest: its page must exist, its character range must fit the page text, and every visual asset ID must resolve to a screenshot or image from the same page. This check runs for a complete `PaperFlowResult` and when a stored chunk set is rehydrated independently.

## Artifact Versioning

`PaperChunkSet.producer` records splitter identity, installed splitter version, chunk size, and overlap. Its members have contiguous positions and must all point back to the artifact and source revision.

`PaperGlobalSummary.producer` records:

- model identity;
- prompt version;
- manager/research-agent orchestration version;
- exact input chunk-set ID;
- coordinator/research instructions hash;
- coordinator output limit;
- research call, concurrency, turn, and per-worker output limits.

Changing any producer field creates a distinct artifact ID. Multiple chunk sets and summaries may coexist for one source revision. Loading a complete `PaperFlowResult` without explicit artifact IDs is allowed only when one unambiguous linked pair exists.

## Citation and Lineage Integrity

A `PaperCitation` identifies the exact chunk set, chunk, page, and optional verbatim quote. `PaperFlowResult` rejects citations to missing chunks, pages outside the cited chunk spans, or quotes absent from chunk text.

`PaperGlobalSummary.derived_from` contains `ArtifactLocator` values. At least one locator must point to its producer's exact input chunk set, with the same source revision and no member ID. The library stores this relationship explicitly so lineage can be checked independently from the summary JSON.

## Retrieval Boundary

Canonical paper models do not implement `embedding_text()` and do not select retrieval text. `quantmind.library` projects:

- one text-embedding target for the global summary;
- one text-embedding target per paper chunk;
- no aggregate target for a chunk set.

`ArtifactLocator` addresses a source revision, artifact, artifact kind, and optional member. The optional source revision keeps the locator usable for legacy `BaseKnowledge` results; V1 paper locators always set it. `LocalKnowledgeLibrary.resolve()` returns the canonical summary, chunk set, chunk, knowledge item, or tree node selected by a locator.

`SearchProjection` is separate from the locator. It records the rebuildable projection kind, version, modality, model, dimensions, and content hash that produced a ranked `SemanticHit`.

## Compatibility Boundary

`LegacyPaper` retains the pre-V1 `TreeKnowledge` shape only so existing version-2 databases and the bundled legacy example can be opened. It is not exported as `Paper`, is not produced by `paper_flow`, and is not part of the V1 paper contract.

There is no `PaperTree` in V1. A future tree or PageIndex artifact must be independently versioned, linked to an exact source and inputs, and added only with a real retrieval use case and migration plan.
