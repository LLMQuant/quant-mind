# Store and search canonical knowledge locally

## Quick Summary

- **Purpose**: Define how the local library stores validated knowledge and source-first papers, then searches rebuildable projections.
- **Read when**: Changing `LocalKnowledgeLibrary`, SQLite schema, semantic projections, filters, hit locators, or file ownership.
- **Status**: Implemented by `quantmind.library` with SQLite schema version 4 and private LlamaIndex ranking.
- **Core rule**: Canonical sources and artifacts are durable; projection text, vectors, and private indexes are rebuildable.
- **User guide**: [`docs/library.md`](../../../docs/library.md)

## Contents

- [Public Contract](#public-contract)
- [Ownership](#ownership)
- [Canonical Storage](#canonical-storage)
- [Paper Transaction](#paper-transaction)
- [Search Projections](#search-projections)
- [Search and Resolution](#search-and-resolution)
- [Selective Rebuild](#selective-rebuild)
- [Financial Time](#financial-time)
- [Integrity and Migration](#integrity-and-migration)
- [PageIndex Boundary](#pageindex-boundary)
- [Out of Scope](#out-of-scope)

## Public Contract

`LocalKnowledgeLibrary` is the only public backend class. Its common operations are:

| Operation | Contract |
|---|---|
| `open()` | Open or migrate a SQLite library without network I/O. |
| `put()` | Store one conventional `BaseKnowledge` item and its required projections. |
| `put_paper()` | Store one `PaperFlowResult`, including exact source assets, two artifacts, lineage, and required projections. |
| `put_paper_structure_tree()` | Atomically store one exact source revision and validated structure tree without projections or embeddings. |
| `get()` | Rehydrate one conventional knowledge item. |
| `get_paper()` | Rehydrate one unambiguous source/chunk-set/summary result, or use explicit artifact IDs when versions coexist. |
| `get_artifact()` | Rehydrate a paper chunk set, global summary, or structure tree by artifact ID. |
| `search()` | Filter and rank rebuildable projections, returning `SemanticHit` evidence. |
| `resolve()` | Resolve a hit locator to its canonical aggregate or member. |

The public types are `LocalKnowledgeLibrary`, `SemanticQuery`, `SemanticHit`, and `SearchProjection`. SQLite rows, embedding providers, LlamaIndex nodes, retrievers, and indexes remain private.

## Ownership

| Owner | Responsibility |
|---|---|
| `quantmind.knowledge` | Define immutable canonical models; perform no I/O and choose no retrieval text. |
| `quantmind.library` | Persist canonical values, own retrieval projections, maintain vectors, and resolve search evidence. |
| [`quantmind.rag`](../rag/document.md) | Build page-aware chunks or transient document-local evidence without owning canonical persistence. |
| `quantmind.flows` | Produce validated results and leave persistence explicit. |
| Caller | Choose database and temporary parser-artifact locations and manage application lifecycle. |

For conventional `BaseKnowledge`, source references remain pointers and the caller retains external raw files. For `PaperFlowResult`, the exact source PDF, screenshots, and extracted images are part of the durable source revision and are copied into the library transaction.

## Canonical Storage

SQLite schema version 4 keeps conventional and source-first paper storage explicit and permits vectorless paper artifacts.

Conventional knowledge uses:

- `knowledge_items` for canonical typed aggregate payloads;
- `knowledge_nodes` for normalized tree members;
- `semantic_records` for rebuildable conventional projections and vectors.

Source-first papers use:

- `paper_sources` for immutable source-revision metadata and canonical manifests;
- `paper_source_assets` for linked content-addressed bytes and asset metadata;
- `paper_artifacts` for independently versioned chunk sets, summaries, and structure trees;
- `paper_artifact_members` for directly addressable chunks or structure nodes;
- `paper_artifact_lineage` for derived-artifact input relationships such as summary-to-chunk-set provenance;
- `paper_projections` for rebuildable summary and chunk text embeddings.

There is no single opaque paper JSON blob and no canonical vector field. Aggregate JSON and normalized relationship rows are cross-checked during rehydration.

## Paper Transaction

`put_paper()` first validates the complete `PaperFlowResult`, computes its source and artifact canonical forms, determines affected projections, and obtains every required embedding. Only then does it begin one `BEGIN IMMEDIATE` transaction.

The transaction writes or reuses the source, asset blobs, artifacts, members, lineage, and all required projections. Any constraint, integrity, or write failure rolls back the transaction. An embedding-provider failure occurs before the transaction and therefore leaves no partial source or artifact rows.

Putting an unchanged result is idempotent and reuses valid vectors. The same source may own multiple chunk-set and summary versions. Artifact identity includes producer configuration, so one version never silently replaces another.

`put_paper_structure_tree(source, tree)` is a separate vectorless transaction. It validates that every node citation belongs to the exact source, writes or reuses the source and its assets, then stores the tree and normalized members. It does not require `put_paper()`, a chunk set, an embedding provider call, or artifact-lineage rows.

## Search Projections

Retrieval text selection lives in `quantmind.library._internal.retrieval_targets`, not canonical models.

- Each supported flat knowledge item produces one whole-item projection.
- Each `TreeKnowledge` produces one aggregate projection and one projection for every non-root node.
- A paper global summary produces one aggregate projection.
- Every paper chunk produces one member projection.
- A paper chunk-set aggregate is not searchable by itself.

Every stored projection records target identity, exact projection text and hash, projection schema version, canonical schema version, source content hash, embedding model, vector dimension, and bytes. Search returns the exact projected text as `SemanticHit.matched_text`.

## Search and Resolution

`SemanticQuery.artifact_kinds` is a Pydantic-validated list of `PaperArtifactKind` values rather than an open string list. `PaperArtifactKind.GLOBAL_SUMMARY` selects summary aggregates and `PaperArtifactKind.CHUNK_SET` selects chunk members owned by chunk-set artifacts. JSON and YAML callers may still use the enum values `paper_summary` and `paper_chunk_set`; unknown values fail validation. Existing `item_types`, source, confidence, tag, tree, `as_of`, and `available_at` filters continue to apply where meaningful. Filtering happens before ranking.

Each `SemanticHit` carries two complementary values:

- `locator`: source revision, artifact ID, artifact kind, and optional member ID needed to resolve canonical evidence;
- `projection`: projection version, modality, embedding model, dimensions, and projection content hash used for ranking.

For paper hits, `search()` rehydrates and validates the owning source and artifact before returning evidence. Summary citations come from the canonical summary. Chunk citations are reconstructed from page and character spans. Callers pass the locator to `resolve()` to obtain a `PaperGlobalSummary` or exact `PaperChunk`.

Compatibility fields `item_id`, `node_id`, and `item_type` remain on `SemanticHit` for conventional knowledge callers. For paper hits they mirror artifact ID, member ID, and artifact kind.

## Selective Rebuild

A projection is reusable only when all recorded identities still match: projection hash and schema version, canonical schema version, source content hash, embedding model, and requested vector dimension. Invalid vector bytes or dimensions are rejected.

`put()` and `put_paper()` send only affected target text to the embedding provider. Changing summary prose or producer configuration does not re-embed unchanged chunks. Closing and reopening the library loads persisted vectors without document re-embedding; a search embeds only the query.

The private LlamaIndex retrieval state is rebuilt lazily after open or mutation. It is never canonical state and may be replaced without migrating source or artifact payloads.

## Financial Time

`as_of` is the latest date covered by knowledge. `available_at` is when the exact source version became observable. `available_at_before` excludes records with a later or unknown availability time; filtering only on `as_of_before` may still leak future information.

Paper projections inherit both times from their exact source revision. This prevents a later summary run from changing the historical availability of the underlying paper.

## Integrity and Migration

Open performs an explicit SQLite user-version migration from schema 2 to schema 3 by adding paper tables and indexes without rewriting conventional knowledge. Older canonical class references for the pre-V1 paper tree load as `LegacyPaper` solely for database compatibility. Schema 3 to 4 permits zero-projection artifacts and adds hierarchical member parents without rewriting conventional knowledge.

Reads fail closed when canonical hashes, counts, IDs, membership, lineage, source relationships, vector bytes, or asset metadata disagree. Blob SHA-256 hashes and byte lengths are checked, and stored asset table fields must match the canonical source manifest. Missing IDs raise `KeyError`; stale or corrupt linked state raises `RuntimeError` with context.

## PageIndex Boundary

The library is canonical storage plus collection-wide semantic search, not a vector database abstraction. `PaperStructureTree` is persisted as a zero-projection artifact, and a node locator resolves to content assembled directly from its cited canonical source pages. Reasoning-based retrieval is owned by `quantmind.mind`. See [Build and retrieve from a page-preserving structure tree](../mind/retrieval.md).

PageIndex-style retrieval is not required to use `LocalKnowledgeLibrary.search()` or private LlamaIndex vector ranking. Paper Flow V1 still returns chunks and a cited summary; structure construction and persistence are separate explicit operations. Building per-node projections is an explicit P2 step that will enable hybrid semantic-plus-agentic retrieval without a second index.

## Out of Scope

- a public storage, vector-store, retriever, or provider hierarchy;
- implicit persistence inside flows;
- answer synthesis or agent memory;
- merging distinct canonical identities;
- per-node structure projections or implicit hybrid seeding;
- treating rebuildable projections as canonical knowledge.
