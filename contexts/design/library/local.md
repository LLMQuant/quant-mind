# Local Semantic Knowledge Library Design

## Quick Summary

- **Purpose**: Define local persistence and semantic retrieval for validated QuantMind knowledge.
- **Read when**: Changing `LocalKnowledgeLibrary`, indexing, semantic filters, retrieval identity, or storage ownership.
- **Status**: Current design owned at runtime by `quantmind.library`.
- **Core boundary**: Canonical knowledge is source truth, semantic records and vectors are derived indexes, and raw source artifacts remain caller-owned.
- **User guide**: [`docs/library.md`](../../../docs/library.md)

## Contents

- [Decision Summary](#decision-summary)
- [Ownership Boundaries](#ownership-boundaries)
- [Canonical and Derived Data](#canonical-and-derived-data)
- [Retrieval Grain and Identity](#retrieval-grain-and-identity)
- [Derived-Index Invalidation](#derived-index-invalidation)
- [Financial-Time Semantics](#financial-time-semantics)
- [Local Ranking Choice](#local-ranking-choice)
- [Non-goals](#non-goals)

## Decision Summary

`quantmind.library` owns persistence and semantic retrieval for canonical
QuantMind knowledge. It replaces the obsolete repository concept of a generic
storage layer that stored raw files, knowledge JSON, embeddings, and indexes
behind one extensible backend abstraction.

The current public surface is deliberately domain-level:

- `LocalKnowledgeLibrary`
- `SemanticQuery`
- `SemanticHit`

There is no public `Storage`, `VectorStore`, `Retriever`, provider registry, or
backend hierarchy. A new backend abstraction is justified only by a second
real implementation and a stable shared contract.

## Ownership Boundaries

| Layer | Responsibility |
|---|---|
| `quantmind.knowledge` | Immutable canonical schemas and embedding projections; no I/O |
| `quantmind.library` | Persist canonical knowledge, maintain rebuildable semantic records, and return typed evidence |
| `quantmind.flows` | Produce canonical knowledge and optionally pass it to a library consumer |
| `quantmind.mind` or an agent application | Use retrieval as a tool and synthesize answers |
| Caller or source-specific pipeline | Retain raw PDF, HTML, media, and operational artifacts |

Raw source retention is intentionally outside the V1 library. A canonical
`SourceRef` and citations preserve provenance identity; they do not turn the
library into an artifact store.

## Canonical and Derived Data

Canonical `BaseKnowledge` is the source of truth. Embeddings, projection text,
filter columns, and vector indexes are derived and rebuildable even when they
share one SQLite database.

The local implementation separates three concerns:

- `knowledge_items` stores one validated aggregate root per knowledge item.
- `knowledge_nodes` stores canonical `TreeNode` values separately with parent,
  position, content hash, and item ownership.
- `semantic_records` stores item/root/node projections and vector metadata used
  by exact cosine ranking.

Concrete types do not get a table per Pydantic subtype. The aggregate-root plus
normalized-node design preserves typed reconstruction while giving future tree
navigation a node-level persistence boundary.

## Retrieval Grain and Identity

- A `FlattenKnowledge` item produces one semantic target from its exact
  `embedding_text()` projection.
- A `TreeKnowledge` item produces one item target for its root with
  `node_id=None`, plus one target for every non-root node.
- Target identity distinguishes a whole item from one of its nodes.
- `SemanticHit.matched_text` is the exact projection used for ranking.
- Callers use `get(item_id)` to resolve full canonical knowledge and node paths.

Re-putting unchanged knowledge with the same canonical item ID is idempotent
and reuses its embeddings. This does not claim deduplication across independent
extraction runs that generated different canonical IDs.

## Derived-Index Invalidation

Every stored vector records the information needed to decide whether it is
reusable: embedding model, dimension, projection hash, source content hash,
knowledge schema version, and projection schema version.

Changed metadata invalidates only affected targets. Canonical deletion removes
the item, its normalized nodes, and its derived semantic records in one
transaction. Corrupt dimensions, vector bytes, canonical payloads, or orphaned
derived records fail explicitly instead of producing a plausible partial hit.

## Financial-Time Semantics

`as_of` and `available_at` answer different questions:

- `as_of` is the information cutoff represented by the knowledge.
- `available_at` is when that source version became observable.

`available_at_before` excludes records whose availability is unknown or after
the cutoff. `as_of_before` alone does not prevent look-ahead. Source kind,
item type, confidence, tags, tree ID, and both time cutoffs combine before
ranking.

## Local Ranking Choice

SQLite provides transactions, foreign keys, typed reconstruction, and explicit
stale/corrupt behavior for canonical knowledge. NumPy exact cosine ranking is
sufficient for the current local scale and keeps the derived search layer
replaceable without changing user code.

A future approximate or remote index may replace the private derived layer. It
does not replace canonical persistence and must not leak a provider-specific
result through `SemanticHit`.

## Non-goals

- raw source artifact storage;
- a generic RAG framework or answer synthesis;
- a public embedder, vector-store, retriever, or backend registry;
- PageIndex tree construction or navigation;
- cross-run knowledge deduplication without a separate stable-ID contract.
