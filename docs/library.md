# Local Semantic Knowledge Library

`quantmind.library` persists canonical `BaseKnowledge` in SQLite and ranks
rebuildable NumPy indexes with exact cosine similarity. It is a financial
knowledge API, not a generic RAG framework: provider and storage details remain
private, and search returns typed QuantMind evidence without generating an
answer.

## Retrieval grain

- Each `FlattenKnowledge` item produces one target from its exact
  `embedding_text()` projection.
- Each `TreeKnowledge` produces one item target for the root with
  `node_id=None`, plus one target for every non-root `TreeNode`.
- Canonical JSON is the source of truth. Embeddings and filter columns are
  derived data and can be replaced by re-putting the canonical item.

The durable vector metadata records the embedding model and dimension, exact
projection hash, source content hash, knowledge schema version, and projection
schema version. Re-putting an unchanged item ID reuses its vectors. A changed
node projection replaces only that node vector; model, dimension, source hash,
or schema changes replace the affected item's vectors.

## Financial-time filters

`as_of` is the information cutoff represented by the knowledge. `available_at`
is when its source became observable. They intentionally remain separate:

- `as_of_before` includes records whose information cutoff is at or before the
  query cutoff.
- `available_at_before` includes only records with a known availability time at
  or before the query cutoff. Unknown availability is excluded.

Consequently, `as_of_before` alone does not prevent look-ahead. Ingestion flows
should populate `available_at` from publication time. When publication time is
unknown, the caller can copy `SourceRef.fetched_at` to `available_at` as a
conservative upper bound.

`item_types` and `source_kinds` use any-of matching. Every requested tag must be
present. `confidence`, `tree_id`, and both time cutoffs combine with those
filters before ranking.

## Usage

```python
from quantmind.library import LocalKnowledgeLibrary, SemanticQuery

library = await LocalKnowledgeLibrary.open(
    ".quantmind/library.db",
    embedding_model="text-embedding-3-small",
)
try:
    await library.put(paper)
    hits = await library.search(
        SemanticQuery(
            text="management expects capital expenditure to increase",
            available_at_before=research_cutoff,
            top_k=10,
        )
    )
    evidence = await library.get(hits[0].item_id)
finally:
    await library.close()
```

`SemanticHit.matched_text` is the exact projection used in ranking. Use
`get(item_id)` to resolve the full canonical item and, for node hits, the node
identified by `node_id`.

Missing IDs raise `KeyError`. Stored canonical payloads that no longer validate
and orphaned or mismatched derived records raise `RuntimeError` with stale-data
context. Invalid vector bytes and inconsistent stored dimensions raise a clear
corrupt-index `RuntimeError`; provider or query dimension mismatches raise
`ValueError`. `delete()` does not deserialize canonical JSON, so a stale
canonical payload can still be removed together with all derived targets in one
transaction.
