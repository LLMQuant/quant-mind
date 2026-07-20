# Local Semantic Knowledge Library

`LocalKnowledgeLibrary` persists canonical QuantMind values in SQLite and ranks rebuildable text-embedding projections with a private LlamaIndex retriever. The canonical storage, transaction, financial-time, migration, and PageIndex boundaries live in the [local library design](../contexts/design/library/local.md).

## Store a Paper Flow Result

Run `paper_flow()` first, then explicitly store its complete result:

```python
result = await paper_flow(
    ArxivIdentifier(id="1706.03762v7"),
    cfg=PaperFlowCfg(model="gpt-4o-mini"),
)

library = await LocalKnowledgeLibrary.open(
    ".quantmind/library.db",
    embedding_model="text-embedding-3-small",
)
try:
    await library.put_paper(result)
finally:
    await library.close()
```

`put_paper()` persists the exact source PDF and retained parser assets, one page-aware chunk-set artifact, one cited global-summary artifact, explicit lineage, and required summary/chunk projections. It obtains all affected embeddings before opening the SQLite transaction, so an embedding failure leaves no partial paper.

Putting the same result again is safe and reuses valid vectors. A changed splitter or summary producer creates another independently addressable artifact version for the same source.

## Store and Resolve a Structure Tree

After building a self-contained `PaperStructureTree` with `PaperFlow`, persist just the tree — no source, chunk, summary, or embedding projections — then reopen it by id:

```python
tree_flow = PaperFlow(PaperStructureCfg(model="gpt-4o-mini"))
structure = await tree_flow.build(ArxivIdentifier(id="1706.03762v7"))

await library.put(structure)                        # standalone; no source needed
tree = await library.open_structure(structure.id)   # identical self-contained value
```

The structure tree is derived only from the exact source pages and structuring producer configuration. Splitter settings and chunk-set versions do not affect its identity. The tree is **self-contained**: its leaf nodes carry their own page text and it carries its own provenance metadata (`as_of` + a light source ref), so it round-trips through `put()` / `open_structure()` to an identical value and can be persisted and retrieved from with no source or chunk set present. A node `ArtifactLocator` passed to `resolve()` returns a `TreeNode` with its stored `content` (no query-time refill). Building node projections and semantic hybrid seeding are deferred to P2.

## Reopen, Search, and Resolve

Opening a library performs no embedding or network request. Search embeds only the query when stored projections are reusable:

```python
from quantmind.knowledge import PaperArtifactKind

library = await LocalKnowledgeLibrary.open(
    ".quantmind/library.db",
    embedding_model="text-embedding-3-small",
)
try:
    summary_hits = await library.search(
        SemanticQuery(
            text="What is the paper's central contribution?",
            artifact_kinds=[PaperArtifactKind.GLOBAL_SUMMARY],
            top_k=3,
        )
    )
    chunk_hits = await library.search(
        SemanticQuery(
            text="How does multi-head attention work?",
            artifact_kinds=[PaperArtifactKind.CHUNK_SET],
            top_k=5,
        )
    )
    evidence = [
        await library.resolve(hit.locator)
        for hit in (*summary_hits, *chunk_hits)
    ]
finally:
    await library.close()
```

A `paper_summary` hit resolves to `PaperGlobalSummary`. A `paper_chunk_set` hit has a member ID and resolves to the exact `PaperChunk`, including source-page spans. Structure trees are retrieved by reasoning over titles and summaries through `AgenticRetriever(RetrievalCfg(...)).retrieve()` in `quantmind.mind` — an LLM agent traverses the structure — not by semantic search in the vectorless MVP. Every `SemanticHit` also includes:

- `matched_text`, the exact library-owned projection used for ranking;
- `projection`, the projection version, model, dimensions, and content hash;
- source metadata, financial time, and canonical citations;
- compatibility fields `item_id`, `node_id`, and `item_type`.

Use `get_artifact(artifact_id)` when the aggregate ID is already known. Use `get_paper(source_revision_id, chunk_set_id=..., summary_id=...)` to reconstruct a compatible result. Artifact IDs may be omitted only when one unambiguous linked chunk-set/summary pair exists.

The complete runnable path is [examples/flows/paper.py](../examples/flows/paper.py).

## Conventional Knowledge

`put(item)` and `get(item_id)` remain available for supported `BaseKnowledge` values such as `News`, `Earnings`, `Factor`, `Thesis`, and generic trees. Canonical models do not implement `embedding_text()`; library-owned projection rules select searchable text.

`SemanticQuery` supports item type, source kind, confidence, tag, tree, `as_of`, and `available_at` filters. Use `available_at_before` to prevent look-ahead: an `as_of` cutoff alone does not prove that the source was observable at that time.

## Bundled Compatibility Example

The bundled AI-infrastructure scenario contains primary-source-backed `News`, `Earnings`, and one pre-V1 `LegacyPaper` tree. Its canonical JSON is precompiled into a SQLite database with six `text-embedding-3-small` targets, so the example embeds only the query:

```bash
python examples/library/semantic_search.py
```

`LegacyPaper` exists only so older databases and this auditable example remain readable. New paper ingestion uses `PaperFlowResult` and `put_paper()`.

Maintainers can regenerate the bundle after changing source data, projection rules, or the storage schema:

```bash
python scripts/examples/build_ai_infrastructure_bundle.py
```

The bundle's facts and short citations come directly from the [Compute Trends Across Three Eras of Machine Learning paper](https://arxiv.org/abs/2202.05924), [Microsoft's FY2025 AI-datacenter investment announcement](https://blogs.microsoft.com/on-the-issues/2025/01/03/the-golden-opportunity-for-american-ai/), and [NVIDIA's FY2026 Q1 results](https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Announces-Financial-Results-for-First-Quarter-Fiscal-2026/default.aspx).

Missing IDs raise `KeyError`. Canonical payloads, linked rows, or asset metadata that no longer agree raise `RuntimeError` with stale-data context. Invalid vector bytes and inconsistent dimensions raise a corrupt-index `RuntimeError`; provider or query dimension mismatches raise `ValueError`.
