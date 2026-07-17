# Local Semantic Knowledge Library

This page only explains how to run the bundled example. The canonical storage,
retrieval, financial-time, and PageIndex boundaries live in the
[local library design](../contexts/design/library/local.md).

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
