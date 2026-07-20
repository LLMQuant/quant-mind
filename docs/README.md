# QuantMind Component Catalog

This is the discovery index for QuantMind's public operations, supported
sources, examples, design documents, and verification commands. Source-specific
acquisition mechanics remain internal unless they are intentionally documented
as a public preprocessing primitive.

Public callable names follow the
[operation naming contract](../contexts/design/operations/naming.md). The runtime API serves
Python callers; coding-agent guidance lives in the repository development
harness.

## Public Operations

| Operation | Import | Input and config | Result | Example | Design or guide |
|---|---|---|---|---|---|
| Source-first paper flow | `quantmind.flows.paper_flow` | `PaperInput`, `PaperFlowCfg` | `PaperFlowResult` | [Persist and search a paper](../examples/flows/paper.py) | [Paper flow design](../contexts/design/flow/paper.md) |
| Paper structure build | `quantmind.flows.PaperFlow` | `PaperFlow(PaperStructureCfg)`; `build()`: `PaperInput` | `PaperStructureTree` (self-contained) | [Build and retrieve](../examples/mind/paper_structure_retrieval.py) | [Structure retrieval design](../contexts/design/mind/retrieval.md) |
| Reasoning-based retrieval | `quantmind.mind.Retrieve` | `Retrieve(AgenticRetrievalCfg)`; `retrieve()`: one `StructureTree` + question (no library) | `list[RetrievalEvidence]` | [Build and retrieve](../examples/mind/paper_structure_retrieval.py) | [Structure retrieval design](../contexts/design/mind/retrieval.md) |
| News collection | `quantmind.flows.collect_news` | `NewsWindow`, `NewsCollectionCfg` | `NewsBatch` from `quantmind.preprocess` | [Collect news](../examples/flows/collect_news.py) | [News collection design](../contexts/design/flow/news.md) |
| Bounded fan-out | `quantmind.flows.batch_run` | Operation inputs and shared config | `BatchResult` | [README usage](../README.md#-usage-examples) | API docstrings |
| Local semantic search | `quantmind.library.LocalKnowledgeLibrary` | `BaseKnowledge` or `PaperFlowResult`, `SemanticQuery` | `list[SemanticHit]` | [Library example](../examples/library/README.md) | [Library guide](library.md) |
| Page-aware document RAG | `quantmind.rag.chunk_parsed_document`, `quantmind.rag.retrieve_parsed_document` | `ParsedDocument`, splitter config, and query | `tuple[ParsedDocumentHit, ...]` | [Paper RAG](../examples/rag/paper.py) | [Document RAG design](../contexts/design/rag/document.md) |

Import public inputs and configs from `quantmind.configs`, flow operations and
builders from `quantmind.flows`, and cognitive services from `quantmind.mind`.
Import result contracts from the canonical layer shown in the catalog.

## Public-Network Sources

| Source | Source selection | Operation | Live-network component smoke test |
|---|---|---|---|
| PR Newswire | `NewsWindow(source="pr-newswire", ...)` | `collect_news` | `python scripts/verify_news_e2e.py` |
| arXiv Transformer PDF | `ArxivIdentifier(id="1706.03762v7")` | `paper_flow`, persistence, reopen, search, and resolution | `python scripts/verify_pdf_rag_e2e.py` |

The PR Newswire smoke test checks the public RSS feed, a complete preceding
24-hour listing window, and ticker-hint recall on a bounded sample of up to 25
article pages. The `news` job in `.github/workflows/e2e.yml` runs daily, on
manual dispatch, and only on pull requests that change its dependency paths.
It is not a required merge check, so external PR Newswire availability cannot
block unrelated changes.

The `paper-flow` job fetches exact arXiv revision `1706.03762v7`, preserves at
least 15 pages, runs bounded `gpt-4o-mini` summarization, persists summary and
chunk projections with `text-embedding-3-small`, reopens the database, searches
both artifact kinds, and resolves every hit. It runs daily, manually, and on
pull requests that change its dependency paths, and it remains non-required
because arXiv and model providers are public-network dependencies.
When the repository `OPENAI_API_KEY` secret is unavailable, the job emits an
explicit skip notice instead of reporting an implementation failure; the
catalog command remains the direct way to run the same bounded slice locally.

## Verification

Run the deterministic required verification for every change:

```bash
bash scripts/verify.sh
```

It covers formatting, linting, typing, import boundaries, unit tests, and
coverage, and must remain network-free. The required `.github/workflows/ci.yml`
workflow runs this same harness after file-hygiene hooks. When a change affects
a public-network component, also run every applicable live-network smoke test
listed above; `.github/workflows/e2e.yml` owns those component jobs.

## Adding a Public Operation or Source

Use the `quantmind-dev` component workflow. A public operation is not complete
until its typed contract, package exports, offline tests, focused example,
design or guide, and catalog row agree. A public-network source additionally
needs mocked source tests plus a bounded live verifier and component job in
`.github/workflows/e2e.yml`.

Each live component owns one `scripts/verify_<component>_e2e.py` command and
one named job in the existing `e2e.yml`. Extend the workflow's precise PR path
filter for the component. When multiple live jobs exist, use GitHub-native
per-job change detection so only affected component jobs run. Add commands only
to the catalog above; root agent guidance stays component-neutral. Do not
create a workflow per component or a generic E2E runner, registry, or base
class.
