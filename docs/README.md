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
| Paper extraction | `quantmind.flows.paper_flow` | `PaperInput`, `PaperFlowCfg` | `Paper` | [README usage](../README.md#-usage-examples) | [Paper E2E design](../contexts/design/flow/paper.md) |
| News collection | `quantmind.flows.collect_news` | `NewsWindow`, `NewsCollectionCfg` | `NewsBatch` from `quantmind.preprocess` | [Collect news](../examples/flows/collect_news.py) | [News collection design](../contexts/design/flow/news.md) |
| Bounded fan-out | `quantmind.flows.batch_run` | Operation inputs and shared config | `BatchResult` | [README usage](../README.md#-usage-examples) | API docstrings |
| Local semantic search | `quantmind.library.LocalKnowledgeLibrary` | `BaseKnowledge`, `SemanticQuery` | `list[SemanticHit]` | [Library example](../examples/library/README.md) | [Library guide](library.md) |
| Page-aware PDF RAG | `quantmind.preprocess.parse_pdf` | Exact PDF bytes and optional artifact directory | `ParsedDocument` | [Paper RAG](../examples/preprocess/paper_rag.py) | [PDF preprocessing design](../contexts/design/preprocess/pdf.md) |

Import public inputs and configs from `quantmind.configs` and current public
operations from `quantmind.flows`. Import result contracts from the canonical
layer shown in the catalog.

## Public-Network Sources

| Source | Source selection | Operation | Live-network component smoke test |
|---|---|---|---|
| PR Newswire | `NewsWindow(source="pr-newswire", ...)` | `collect_news` | `python scripts/verify_news_e2e.py` |
| arXiv Transformer PDF | `fetch_arxiv("1706.03762v7")` | `parse_pdf` and LlamaIndex retrieval | `python scripts/verify_pdf_rag_e2e.py` |

The PR Newswire smoke test checks the public RSS feed, a complete preceding
24-hour listing window, and ticker-hint recall on a bounded sample of up to 25
article pages. The `news` job in `.github/workflows/e2e.yml` runs daily, on
manual dispatch, and only on pull requests that change its dependency paths.
It is not a required merge check, so external PR Newswire availability cannot
block unrelated changes.

The PDF RAG smoke fetches the pinned 15-page Transformer paper, parses ordered
pages, creates LlamaIndex chunks, and verifies that BM25 retrieval returns a
multi-head-attention passage with page metadata. It does not call an LLM.

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
