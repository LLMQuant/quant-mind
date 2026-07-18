# Orchestration and Construction Altitude

## Quick Summary

- **Purpose**: Decide where construction/identity logic lives, and when a step should be an autonomous agent versus deterministic code.
- **Read when**: Adding or refactoring any flow that builds knowledge models from preprocess/rag values, or that calls a model more than once.
- **Status**: Derived from the Paper Flow V1 refactor. Applies to every flow.
- **Core rule**: Identity belongs in the knowledge layer; translation belongs in the flow. Use an autonomous agent only when the decomposition must be decided at runtime.

## Contents

- [Two Principles](#two-principles)
- [Principle 1: Identity vs Translation Altitude](#principle-1-identity-vs-translation-altitude)
- [Principle 2: Agentic vs Deterministic Orchestration](#principle-2-agentic-vs-deterministic-orchestration)
- [Worked Example: Paper Flow V1](#worked-example-paper-flow-v1)
- [Checklist](#checklist)

## Two Principles

Both principles came from one observation: a flow file had become large and was reaching into another module's private helpers. The cause was not bad code locally — it was two responsibilities placed at the wrong altitude.

1. **Identity vs translation altitude** — who mints IDs and hashes, and who maps one type system onto another.
2. **Agentic vs deterministic orchestration** — whether a model decides *how* the work is decomposed, or code does.

## Principle 1: Identity vs Translation Altitude

"Construction" is two separable concerns that belong in different layers:

| Concern | Depends on | Home |
|---|---|---|
| **Identity** — mint IDs, content/producer hashes, resolve and validate citations | knowledge types only | **knowledge layer** (smart constructors) |
| **Translation** — map `preprocess`/`rag` value objects to knowledge inputs, read asset bytes (IO) | preprocess/rag **and** knowledge | **flow layer** (the only layer that imports both) |

The `knowledge` package is a leaf: an import-linter contract forbids it from importing `preprocess`, `rag`, `flows`, or `library`. So a knowledge model **cannot** name a `ParsedDocument` or `ParsedChunk`, and translation from those types genuinely has to live in the flow. That constraint is real and correct.

But it does **not** justify the flow computing identity. The rule:

- **Knowledge models expose smart constructors** (`from_parsed`, `from_parsed_chunks`, `from_draft`) that take knowledge-native inputs and mint every ID and hash internally. A model that validates its own identity in a `model_validator` must also be able to *construct* it; do not leave construction as free private functions for callers to import.
- **The flow adapts and calls.** It maps the ephemeral, path-based preprocess/rag types into knowledge-native inputs (plain value objects carrying geometry, text, and pre-read bytes) and passes them to the constructors. It imports **zero** private `_paper_*_id` / `_*_hash` helpers and computes no ID itself.

**Smell test.** A module importing another module's underscore-prefixed helpers across a package boundary is almost always identity logic living at the wrong altitude. Move the logic behind a public constructor on the model, not the import into the caller.

## Principle 2: Agentic vs Deterministic Orchestration

The Agents SDK's manager/worker pattern (a coordinator agent that calls other agents via `Agent.as_tool()` and decides how many calls to make) earns its cost only when **the decomposition is decided at runtime** — when the agent must look at intermediate results and choose what to do next.

Ask one question:

> Is the decomposition something code can compute now, or must a model decide it after seeing intermediate results?

- **Code can compute it now** → deterministic map-reduce. Tile the work in code, fan out stateless agents with `asyncio.gather` + a `Semaphore`, run one reducer. Coverage is guaranteed by construction; bounds come from the SDK (`ModelSettings.max_tokens`, `output_type`) and `asyncio` (`wait_for`). No coordinator, no coverage police, no token accountant.
- **A model must decide at runtime** → agentic coordinator with `as_tool`. And in that case do *not* also bolt a deterministic "must cover everything" check on top; that check contradicts the autonomy you just paid for.

**The incoherent middle to avoid.** Running an autonomous coordinator *and* then enforcing a deterministic budget/coverage accountant pays for nondeterminism and immediately suppresses it — both costs, neither benefit. If you find yourself writing a concurrency-safe accountant to make an agent behave deterministically, the task wanted deterministic orchestration from the start.

Using the SDK does **not** force the agentic pattern. Single-agent `Runner.run(agent, output_type=...)` gives structured output, tracing, retries, and guardrails inside a fully deterministic map-reduce — you do not drop to a raw completion API to get determinism.

**Do not build a framework for it.** The fan-out primitive is a function, not a base class: `asyncio.gather` over code-computed items with a `Semaphore`. Keep it inline until a second flow needs it; only then extract a small `map_reduce(items, map_fn, reduce_fn, *, concurrency)` helper. A `BaseFlow`/`ParallelWorkflow` base class is the framework this project explicitly avoids.

## Worked Example: Paper Flow V1

- **Identity** lives on `PaperSourceRevision.from_parsed`, `PaperChunkSet.from_parsed_chunks`, and `PaperGlobalSummary.from_draft`. `quantmind/flows/paper.py` imports no `_paper_*` helper and mints no ID.
- **Translation** stays in the flow: `_fetch_paper_source` (IO), `_adapt_pages` (reads screenshot/image bytes, maps to `PaperPageInput`), `_adapt_chunks` (maps `ParsedChunk` to `PaperChunkInput`).
- **Summarization** is a deterministic map-reduce (`_chunk_groups` tiles the chunk set; one research agent per group; one reducer). The earlier manager/worker coordinator plus its `_SummaryBudget` concurrency accountant were removed — see [Bounded Model Calls](../flow/paper.md#bounded-model-calls).

## Checklist

When adding a flow that turns preprocess/rag values into knowledge models:

- [ ] Does the flow import any `_`-prefixed helper from `knowledge`? If so, move that logic into a smart constructor.
- [ ] Does each knowledge artifact have a `from_*` constructor that mints its own IDs/hashes?
- [ ] Is the flow's remaining work limited to fetch, parse, read bytes, and structural mapping?
- [ ] For multi-call model steps: can code compute the decomposition? If yes, use map-reduce, not a coordinator.
- [ ] Are you enforcing coverage/budget on an autonomous agent? If yes, prefer deterministic decomposition instead.
- [ ] Is the fan-out a plain function (`gather` + `Semaphore`), not a base class?
