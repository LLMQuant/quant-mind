# qm_mcp engineering log

Append-only record of notable changes to the `qm_mcp/` research-corpus layer
(Thomas's additive layer on top of LLMQuant/quant-mind). Upstream `quantmind/`
history lives in the normal git log.

## 2026-06-12 — Phase 4 landing: qm_mcp merged to master

- **PR [#1](https://github.com/AdairBear/quant-mind/pull/1)** squash-merged →
  `9b8a9599d5e00f61f9b2c2e883a02ecf1b0aa90c`.
- Adds the persistence + embedding + semantic-query + MCP layer
  (`store.py`, `embed.py`, `ingest.py`, `query.py`, `server.py`, `cli.py`,
  `seed_corpus.txt`, `_smoke_mcp.py`) that QuantMind v0.2 does not yet ship.
- Companion hermes-agent side: PR
  [#10](https://github.com/AdairBear/hermes-agent/pull/10) →
  `84314fa7eec991eccea8a59024c79f3cef53efbc` (the `#research` channel router +
  `docs/quantmind_brain_boundary.md`).
- Landed in a **new private** `AdairBear/quant-mind` repo (origin left pointing
  at upstream `LLMQuant/quant-mind`; `fork` remote added).
- Verified: direct stdio MCP call enumerates all 7 tools and `qm_query` returns
  grounded, cited answers; corpus live (33 items incl. Databento
  futures-microstructure articles). Live-gateway pickup pending an operator
  restart (see `quantmind_brain_boundary.md` in hermes-agent for the open item).
