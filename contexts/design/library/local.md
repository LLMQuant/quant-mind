# Local Knowledge Library Design

## Quick Summary

- **Purpose**: Define how the local library stores validated knowledge and searches it by meaning.
- **Read when**: Changing `LocalKnowledgeLibrary`, search records, filters, query results, or file ownership.
- **Status**: Current design owned at runtime by `quantmind.library`.
- **Core rule**: Stored `BaseKnowledge` is the data to keep. Search text and vectors can be rebuilt. The caller stores raw PDF, HTML, and media files.
- **User guide**: [`docs/library.md`](../../../docs/library.md)

## Contents

- [Key Decisions](#key-decisions)
- [Who Owns What](#who-owns-what)
- [Stored Knowledge and Rebuildable Search Data](#stored-knowledge-and-rebuildable-search-data)
- [What Can Match a Query](#what-can-match-a-query)
- [When to Rebuild Search Data](#when-to-rebuild-search-data)
- [Time Fields and Look-Ahead](#time-fields-and-look-ahead)
- [Why SQLite and LlamaIndex Ranking](#why-sqlite-and-llamaindex-ranking)
- [Independent tree navigation](#independent-tree-navigation)
- [Out of Scope](#out-of-scope)

## Key Decisions

`quantmind.library` stores validated QuantMind knowledge and searches it by
meaning. It replaces an older design for one generic storage layer that would
hide raw files, knowledge JSON, embeddings, and indexes behind one backend API.

The public API is intentionally small:

- `LocalKnowledgeLibrary`
- `SemanticQuery`
- `SemanticHit`

There is no public `Storage`, `VectorStore`, `Retriever`, provider registry, or
backend class hierarchy. Add a shared backend API only after a second working
implementation proves which behavior is truly shared.

## Who Owns What

| Package or caller | Responsibility |
|---|---|
| `quantmind.knowledge` | Define immutable knowledge models and the text used for embeddings; perform no I/O |
| `quantmind.library` | Store validated knowledge, maintain rebuildable search records, and return `SemanticHit` results |
| [`quantmind.rag`](../rag/document.md) | Chunk and retrieve evidence within one parsed document without storing canonical knowledge |
| `quantmind.flows` | Produce validated knowledge and optionally pass it to a library |
| `quantmind.mind` or an agent application | Search the library and use matches to write answers |
| Caller or source-specific pipeline | Retain raw PDF, HTML, media, and operational files |

The V1 library does not store raw source files. `SourceRef` and citations point
back to the source, but they do not contain the source file itself.

## Stored Knowledge and Rebuildable Search Data

Validated `BaseKnowledge` is the record that must be preserved. Embeddings,
the text sent to the embedder, filter columns, and vector indexes can all be
rebuilt, even when they share one SQLite database with the knowledge records.

The local implementation separates three concerns:

- `knowledge_items` stores one complete validated knowledge item.
- `knowledge_nodes` stores each `TreeNode` in a separate row with its parent,
  position, content hash, and owning item.
- `semantic_records` stores searchable text and vector details for whole items,
  roots, and nodes.

Concrete types do not get a table per Pydantic subtype. Storing each complete
item plus separate node rows lets the library rebuild the original typed item
and later navigate a tree one node at a time.

## What Can Match a Query

- A `FlattenKnowledge` item produces one searchable record from its exact
  `embedding_text()` value.
- A `TreeKnowledge` item produces one record for the whole item with
  `node_id=None`, plus one record for every non-root node.
- Each record says whether it represents a whole item or one node.
- `SemanticHit.matched_text` is the exact text used to rank the match.
- Callers use `get(item_id)` to load the full knowledge item and node paths.

Putting unchanged knowledge with the same item ID is safe to repeat and reuses
its embeddings. The library does not merge separate extraction runs that
created different item IDs.

## When to Rebuild Search Data

Every stored vector records the information needed to decide whether it is
reusable: embedding model, dimension, embedding-text hash, source content hash,
knowledge model version, and embedding-text format version.

When any of those values changes, rebuild only the affected search records.
Deleting an item removes the item, its nodes, and its search records in one
transaction. Invalid vector sizes or bytes, unreadable knowledge data, and
search rows without an item raise an error instead of returning an incomplete
match that looks valid.

## Time Fields and Look-Ahead

`as_of` and `available_at` answer different questions:

- `as_of` is the latest date covered by the knowledge.
- `available_at` is when that source version became observable.

`available_at_before` excludes records that became available after the cutoff
or have no known availability time. Filtering only by `as_of_before` can still
leak future information. Apply source kind, item type, confidence, tags, tree
ID, and both time cutoffs before ranking results.

## Why SQLite and LlamaIndex Ranking

SQLite provides transactions, foreign keys, and reliable reconstruction of
typed knowledge. LlamaIndex owns the private collection-wide vector retrieval
and ranking mechanics. On the first search after open or a write, private
retrieval state is rebuilt from the filtered semantic records stored in SQLite;
unchanged records reuse their persisted embeddings and are not sent to the
embedding provider again.

LlamaIndex nodes and retrievers remain implementation details. They do not
enter `SemanticQuery`, `SemanticHit`, canonical Pydantic payloads, or public
signatures. A future approximate or remote index may replace the private
search implementation without replacing stored knowledge or changing the
public result type.

## Independent tree navigation

`LocalKnowledgeLibrary` is canonical knowledge storage with rebuildable
retrieval capabilities; it is not defined as a vector database. A future
PageIndex path can select a paper through collection-wide semantic retrieval,
then navigate that selected document's tree through a separate operation and
separately rebuildable state. PageIndex does not have to be served through
`search()` or LlamaIndex ranking. Opinionated document retrieval, including a
future PageIndex adapter, belongs under [`quantmind.rag`](../rag/document.md).

## Out of Scope

- raw source file storage;
- a generic framework for retrieving context and writing answers;
- a public embedder, vector-store, retriever, or backend registry;
- PageIndex tree construction or navigation;
- merging knowledge across runs without a separate stable-ID design.
