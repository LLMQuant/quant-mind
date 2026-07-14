# QuantMind Component Catalog

This is the discovery index for QuantMind's public operations, supported
sources, examples, design documents, and verification commands. Source-specific
acquisition mechanics remain internal unless they are intentionally documented
as a public preprocessing primitive.

Public callable names follow the
[operation naming contract](design/en/operations.md). The runtime API serves
Python callers; coding-agent guidance lives in the repository development
harness.

## Public Operations

| Operation | Import | Input and config | Result | Example | Design or guide |
|---|---|---|---|---|---|
| Paper extraction | `quantmind.flows.paper_flow` | `PaperInput`, `PaperFlowCfg` | `Paper` | [README usage](../README.md#-usage-examples) | [Papers](papers.md) |
| News collection | `quantmind.flows.collect_news` | `NewsWindow`, `NewsCollectionCfg` | `NewsBatch` from `quantmind.preprocess` | [Collect news](../examples/flows/collect_news.py) | [News collection design](design/en/news.md) |
| Bounded fan-out | `quantmind.flows.batch_run` | Operation inputs and shared config | `BatchResult` | [README usage](../README.md#-usage-examples) | API docstrings |

Import public inputs and configs from `quantmind.configs` and current public
operations from `quantmind.flows`. Import result contracts from the canonical
layer shown in the catalog.

## Public-Network Sources

| Source | Source selection | Operation | Live component gate |
|---|---|---|---|
| PR Newswire | `NewsWindow(source="pr-newswire", ...)` | `collect_news` | `python scripts/verify_news_e2e.py` |

The PR Newswire gate checks the public RSS feed and a complete preceding
24-hour listing window without fetching article pages. Its GitHub workflow
runs on pull requests, daily, and on manual dispatch.

## Verification

Run the deterministic offline golden gate for every change:

```bash
bash scripts/verify.sh
```

It covers formatting, linting, typing, import boundaries, unit tests, and
coverage, and must remain network-free. When a change affects a public-network
component, also run every applicable live gate listed above.

## Adding a Public Operation or Source

Use the `quantmind-dev` component workflow. A public operation is not complete
until its typed contract, package exports, offline tests, focused example,
design or guide, and catalog row agree. A public-network source additionally
needs mocked source tests plus a bounded live verifier and CI workflow.
