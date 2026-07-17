# Local Semantic Knowledge Library

`quantmind.library` persists canonical `BaseKnowledge` in SQLite and ranks
rebuildable semantic records through a private LlamaIndex vector index. It is a financial
knowledge API, not a generic RAG framework: provider and storage details remain
private, and search returns typed QuantMind evidence without generating an
answer.

## Retrieval grain

- Each `FlattenKnowledge` item produces one target from its exact
  `embedding_text()` projection.
- Each `TreeKnowledge` produces one item target for the root with
  `node_id=None`, plus one target for every non-root `TreeNode`.
- Canonical typed records are the source of truth. Embeddings and filter
  columns are derived data and can be replaced by re-putting the item.

## Local storage model

The default local database is SQLite, with separate canonical and derived
concerns:

- `knowledge_items` stores one aggregate root per `BaseKnowledge`. The type
  discriminator and schema version select the concrete Pydantic model.
- `knowledge_nodes` stores every canonical `TreeNode` separately with its item,
  parent, position, payload, and content hash. A large tree is not hidden in the
  aggregate-root JSON row.
- `semantic_records` stores rebuildable item/root/node projections and vectors.

Concrete types do not get tables such as `news`, `earnings`, or `papers`.
Creating one table per Pydantic subtype would duplicate common metadata and
require a database migration whenever the knowledge standard adds a type. The
aggregate-root plus normalized-node model preserves typed validation while
giving future tree navigation a stable node-level storage boundary.

SQLite is the default because canonical knowledge needs transactions, foreign
keys, typed reconstruction, and explicit corrupt/stale/not-found behavior. A
vector database such as Chroma optimizes the derived similarity-search layer;
it does not replace those canonical-storage responsibilities. The current local
implementation rebuilds a private in-memory LlamaIndex vector index from
`semantic_records`, without adding another service. If corpus size later
requires an approximate or remote vector index, that derived layer can change
privately without changing `LocalKnowledgeLibrary`, the canonical tables, or
user code. A future PageIndex operation may navigate a selected document tree
separately; the library is not defined as vector-only.

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

The bundled AI-infrastructure scenario contains primary-source-backed `News`,
`Earnings`, and a real research `Paper` tree. Its canonical source JSON is
precompiled into a SQLite database with six `text-embedding-3-small` targets,
so the user-facing example does not perform ingestion or document embedding.
Put `OPENAI_API_KEY` in `.env` and run:

```bash
python examples/library/semantic_search.py
```

The example:

1. Opens the ready-to-search SQLite bundle through `LocalKnowledgeLibrary`.
2. Embeds only the user's query with `text-embedding-3-small`.
3. Searches a concrete AI-infrastructure question with a no-look-ahead
   availability cutoff.
4. Resolves every hit with `get()` and prints source, citation, financial time,
   and the root-to-node path and content for tree evidence.

Maintainers can regenerate the model-specific database from the auditable JSON
after changing the source data or storage schema:

```bash
python scripts/examples/build_ai_infrastructure_bundle.py
```

The bundle's facts and short citations come directly from the
[Compute Trends Across Three Eras of Machine Learning paper](https://arxiv.org/abs/2202.05924),
[Microsoft's FY2025 AI-datacenter investment announcement](https://blogs.microsoft.com/on-the-issues/2025/01/03/the-golden-opportunity-for-american-ai/),
and [NVIDIA's FY2026 Q1 results](https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Announces-Financial-Results-for-First-Quarter-Fiscal-2026/default.aspx).

Representative output has this shape; scores depend on the selected embedding
model:

```text
Bundle: .../examples/library/data/ai_infrastructure.db
Query: What evidence shows demand for AI infrastructure is expanding?

1. score=0.812 type=paper
   path: ... > ...
   matched: ...
   source: https://...
   citation: ...
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
