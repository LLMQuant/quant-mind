# qm_mcp — QuantMind research-corpus surface

This package turns [QuantMind](../README.md) into a **queryable research
corpus** for Thomas's trading + AVST work, exposed over MCP so Personal
Hermes, Dispatch sessions, the Conductor, and future Akazi AVST all read the
same knowledge base.

## Why this exists

QuantMind v0.2 ships **ingestion + LLM extraction only** — `paper_flow`
fetches an arXiv id / URL / PDF / raw text, converts it to markdown, and
extracts a typed `Paper` tree. Its persistence, embedding, semantic-query,
and "Data MCP" layers are still **vision / future PRs** (PR6/PR7 per their
README). `qm_mcp` supplies exactly that missing Stage-2 layer:

```
ingest (QuantMind paper_flow)
   → CorpusStore   (~/.quantmind/corpus : one JSON + one vector per item)
   → semantic query (OpenAI embeddings → cosine top-k → grounded answer)
   → MCP server    (qm_ingest_*, qm_query, qm_list_corpus, qm_delete_item)
```

It is dependency-light: it reuses QuantMind's own venv (`openai`, `numpy`,
`pydantic`, `httpx`, `mcp`) and stores everything on the local filesystem.

## Secrets

Loaded from `~/.hermes/.env` at runtime — nothing is hard-coded. Embeddings
and `paper_flow` extraction need a **real platform.openai.com** key. Hermes'
`OPENAI_API_KEY` is an OpenRouter key (`sk-or-…`, no embeddings endpoint), so
`qm_mcp` uses `VOICE_TOOLS_OPENAI_KEY` (the real OpenAI key kept for Whisper)
and forces it for this process only.

## Run the MCP server

```bash
/Users/thomasadair/projects/quant-mind/.venv/bin/python -m qm_mcp.server
```

Registered in Hermes `~/.hermes/config.yaml` under `mcp_servers: quantmind`
(see `docs/quantmind_brain_boundary.md` in the hermes-agent repo).

## CLI (seeding + shell use)

```bash
PY=/Users/thomasadair/projects/quant-mind/.venv/bin/python
$PY -m qm_mcp.cli ingest-arxiv 1105.3115
$PY -m qm_mcp.cli ingest-pdf  ~/papers/foo.pdf
$PY -m qm_mcp.cli ingest-url  https://example.com/article
$PY -m qm_mcp.cli seed        qm_mcp/seed_corpus.txt
$PY -m qm_mcp.cli query       "What does Stoikov say about gamma?"
$PY -m qm_mcp.cli list
$PY -m qm_mcp.cli delete      <item_id>
```

## MCP tools

| Tool | Purpose |
|---|---|
| `qm_ingest_arxiv(arxiv_id)` | Ingest an arXiv paper by id or URL |
| `qm_ingest_url(url)` | Ingest a web page / hosted PDF |
| `qm_ingest_pdf(path)` | Ingest a local PDF / HTML / Markdown file |
| `qm_ingest_text(text, title?)` | Ingest pasted text |
| `qm_query(question, k=5)` | Grounded natural-language answer + top-k sources |
| `qm_list_corpus()` | List all ingested items (metadata) |
| `qm_delete_item(item_id)` | Remove one item |

## Storage

`~/.quantmind/corpus/` (outside both git repos — never committed):
- `items/<id>.json` — record: metadata + flattened context + full Paper tree
- `vectors/<id>.npy` — 1536-dim embedding (aligned by id)
- `ingestion_log.jsonl` — append-only ledger of ingestion events

`id` is a stable hash of the source, so re-ingesting is idempotent (dedup).

## Known QuantMind quirks handled here

- **Strict-schema rejection.** `Agent(output_type=Paper)` fails under OpenAI
  strict structured output (recursive UUID-keyed tree). We pass a non-strict
  `AgentOutputSchema(Paper, strict_json_schema=False)`.
- **No news flow.** QuantMind has `knowledge/news.py` types but no
  `news_flow`. News/blog URLs go through the generic `HttpUrl` → `paper_flow`
  path (trafilatura HTML → markdown → extraction).
- **DOI unsupported.** `paper_flow` raises `NotImplementedError` on DOI
  inputs upstream; use arXiv id or a direct URL.
