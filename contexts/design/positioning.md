# QuantMind Positioning

## Quick Summary

- **Purpose**: State what QuantMind is, who its consumer is, and the two engineering dimensions (context and harness) that everything else — README, slides, skills, decks — derives from. This is the single canonical positioning source; when prose elsewhere disagrees with this page, this page wins.
- **Read when**: Writing or reviewing any outward-facing description of QuantMind (README, a talk, a skill, an issue), or deciding whether a feature belongs to the shipped story or the roadmap.
- **Load next**: For the harness mechanics, [`../dev/harness-engineering.md`](../dev/harness-engineering.md); for operation naming used throughout, [`operations/naming.md`](operations/naming.md).
- **Authority**: Current as of July 2026. This page **supersedes** `docs/superpowers/specs/2026-04-16-quantmind-positioning/` (the April "personal knowledge base / MCP product" framing), which is kept only for history. Do not cite the April spec.

## Contents

- [Positioning and Hero](#positioning-and-hero)
- [Context Engineering](#context-engineering)
- [Harness Engineering](#harness-engineering)
- [The Bet](#the-bet)
- [Evaluation In Design](#evaluation-in-design)
- [LLMQuant Data In Production](#llmquant-data-in-production)
- [Roadmap](#roadmap)

## Positioning and Hero

QuantMind is **an agent-native workbench for financial knowledge extraction**. The primary consumer of this repository is not a human importing a package — it is a coding agent working inside the checkout. You describe the pipeline you want; the agent builds it here, against the repo's contracts, skills, and verification.

Hero line: **"Don't import it. Open it."** The intended first move is `cd quant-mind && claude` (or `codex`), not `pip install`. QuantMind remains a perfectly good importable library — `PaperFlow(cfg).build(input)`, `collect_news`, `batch_run` are real Python operations — but the framing is workbench-first, library-second. The workbench is where an agent turns a loose request into a typed, cited, reproducible extraction pipeline.

Two engineering dimensions structure the whole project — dimensions of one repository, not versions of a product. **Context engineering** turns any source into typed knowledge. **Harness engineering** turns any agent into a domain specialist. The first is what the library does; the second is what the repository is. (The July 2026 poster labels the pair "V1 / V2" as a contrast device; serious prose uses the dimension names, never the version labels.)

## Context Engineering

**Any source → typed knowledge.** This dimension lands unstructured financial content in typed, cited, as-of-correct knowledge that downstream retrieval can trust.

- **Deterministic preprocess** — `fetch` / parse / `format` + `clean` produce source-faithful values with no model in the loop, so the provenance is exact and replayable.
- **Config-driven operations** — `PaperFlow(cfg).build(input)` binds an immutable build config once and applies it per input; `collect_news` collects a replayable source window; `batch_run` fans any operation across a list of inputs under one unified setting. Callers do not write `asyncio.gather` boilerplate.
- **Typed knowledge shapes** — a `Paper` structure **tree** for whole-document artifacts, and flat **cards** for `News` / `Earnings` / `Factor` / `Thesis`. Every artifact is self-contained: it carries its own text, an `as_of` timestamp, and a light source ref, so it persists and time-queries standalone.
- **Retrieval over that knowledge** — `rag/` (deterministic chunking + BM25 / similarity), `library/` (local persistence + meaning-based search), and `mind/` (agentic, reasoning-based retrieval where an LLM decides). Together they serve RAG and Agentic RAG, deep research, and data-MCP serving.

Context engineering is the substance shipped as the NeurIPS 2025 GenAI-in-Finance workshop paper (arXiv:2509.21507).

## Harness Engineering

**Any agent → domain specialist.** Harness engineering is the claim that the repository itself — its contracts, disclosure structure, skills, hooks, and verification — is what upgrades a general coding agent into one that reliably does QuantMind-quality work.

- **Repo-level contracts** — `AGENTS.md` / `CLAUDE.md` state the always-on rules once, in one source, for every agent that opens the repo.
- **Progressive-disclosure `contexts/`** — agent-facing reference pages with a Quick Summary / Contents preview, so an agent loads only the one page a task needs instead of the whole design corpus.
- **Portable skills** — `quantmind-dev` ships today (commit / PR / component-development workflow); `quantmind-best-practice` is planned.
- **Claude + Codex hooks** — shared hook scripts give both agents identical hard guarantees (for example, no verification bypass) without maintaining two copies of a rule's content.
- **Deterministic verify** — `scripts/verify.sh` runs lint + types + import boundaries + tests, fast-failing in a fixed order. CI runs the exact same script, so a green local run means a green PR.

The mechanics — which enforcement layer catches whom, how one rule reaches both Claude and Codex — are detailed in [`../dev/harness-engineering.md`](../dev/harness-engineering.md).

## The Bet

The wager behind harness engineering, stated plainly: **a weak model in a good harness beats a strong model running bare.** If that holds, the durable asset is not the model of the month but the repository that mounts around it — the contracts, contexts, skills, and verification that keep any agent on the rails.

## Evaluation In Design

Evaluation is in the **design phase**. No results are claimed here; two benchmarks are being designed, one per dimension.

- **Knowledge eval** (context engineering) — correctness (EM / F1), citation precision and recall, point-in-time correctness (no look-ahead), and task lift. Protocol anchors under consideration: FinanceBench, FailSafeQA, and RAGAS faithfulness.
- **Harness eval** (harness engineering) — SWE-bench-style paired runs: the same model and the same tasks, run bare versus run inside the mounted repo. Metrics: cost-to-green, pass@1, pass^k across seeds, and wall-clock.

Run instrumentation — tokens, cost, duration, and a verify oracle — already ships. The benchmarks that consume it do not yet exist; treat any evaluation claim as a plan until this section says otherwise.

## LLMQuant Data In Production

**LLMQuant Data is QuantMind in production.** The hosted data platform runs extraction pipelines powered by QuantMind; QuantMind is the open engine, LLMQuant Data is the operated product on top of it.

- The dependency direction is one-way: `llmquant-data` imports `quantmind`, never the reverse. Nothing in this repo may depend on the hosted platform.
- "The knowledge harness for AI-native finance" is **LLMQuant Data's** slogan, not QuantMind's. Do not attach it to QuantMind.

## Roadmap

The following are planned, not shipped. Describe them as roadmap; never present them as current capability.

- `quantmind-best-practice` skill (companion to the shipped `quantmind-dev`).
- SEC / filings collection flow.
- Prediction-market knowledge type.
- Concrete `GraphKnowledge` types (today a placeholder).
- The two evaluation benchmarks described in [Evaluation In Design](#evaluation-in-design).
