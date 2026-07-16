# Local Knowledge Library Example

This example searches an auditable AI-infrastructure knowledge bundle containing
`News`, `Earnings`, and a research `Paper` tree.

## Run

Put `OPENAI_API_KEY` in the repository `.env`, then run from the repository root:

```bash
python examples/library/semantic_search.py
```

The bundled SQLite database already contains six 1536-dimensional
`text-embedding-3-small` document targets. Running the example embeds only the
query, then uses `LocalKnowledgeLibrary.open()`, `search()`, and `get()` to show
ranked evidence and tree paths.

## Files

- `semantic_search.py` is the user-facing query example.
- `data/ai_infrastructure.json` is the canonical, reviewable source bundle.
- `data/ai_infrastructure.db` is the generated SQLite library with canonical
  records and model-specific embeddings. Do not edit it directly.

## Rebuild the bundle

Regenerate the database after changing the source JSON, library schema,
retrieval-target rules, embedding model, or dimension:

```bash
python scripts/examples/build_ai_infrastructure_bundle.py
```

The build validates every source item as canonical QuantMind knowledge and
removes an incomplete database if embedding generation fails.

See [the library guide](../../docs/library.md) for storage, retrieval-grain, and
financial-time semantics.
