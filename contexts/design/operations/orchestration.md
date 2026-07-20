# Orchestration and Construction Altitude

## Quick Summary

- **Purpose**: Decide where construction/identity logic lives, and when a step should be an autonomous agent versus deterministic code.
- **Read when**: Adding or refactoring any flow that builds knowledge models from preprocess/rag values, or that calls a model more than once.
- **Status**: Derived from the Paper Flow V1 refactor. Applies to every flow.
- **Core rule**: Identity belongs in the knowledge layer; translation belongs in the flow. A pipeline is pure processing that returns a complete, self-contained artifact; persistence and retrieval are separate downstream concerns. Use an autonomous agent only when the decomposition must be decided at runtime.

## Contents

- [Three Principles](#three-principles)
- [Principle 1: Identity vs Translation Altitude](#principle-1-identity-vs-translation-altitude)
- [Principle 2: Agentic vs Deterministic Orchestration](#principle-2-agentic-vs-deterministic-orchestration)
- [Principle 3: Pipeline vs Component Altitude](#principle-3-pipeline-vs-component-altitude)
- [Callable Shape: Bind Config, Pass the Operand](#callable-shape-bind-config-pass-the-operand)
- [Worked Example: Paper Flow V1](#worked-example-paper-flow-v1)
- [Checklist](#checklist)

## Three Principles

The first two principles came from one observation: a flow file had become large and was reaching into another module's private helpers. The cause was not bad code locally — it was responsibilities placed at the wrong altitude. The third came from a redesign of structure retrieval, where a "pipeline" had quietly taken on persistence and retrieval that belonged elsewhere.

1. **Identity vs translation altitude** — who mints IDs and hashes, and who maps one type system onto another.
2. **Agentic vs deterministic orchestration** — whether a model decides *how* the work is decomposed, or code does.
3. **Pipeline vs component altitude** — what a finished pipeline owns (pure processing to a complete artifact) versus what stays a downstream concern (persistence, retrieval).

These principles do not require every operation to have the same callable shape — see [Callable Shape](#callable-shape-bind-config-pass-the-operand) below.

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

Likewise, a small service may bind a genuinely shared construction-time thing —
a model policy, a provider seam, or (most often) the immutable **config** that
must stay constant across a batch — without introducing a generic workflow
hierarchy; see [Callable Shape](#callable-shape-bind-config-pass-the-operand).
Canonical knowledge artifacts remain frozen values and do not acquire
`build()`, `retrieve()`, persistence, or provider state merely to make call
sites look object-oriented.

## Principle 3: Pipeline vs Component Altitude

A **pipeline** (in `quantmind.flows`) is a finished, batteries-included workflow: a caller states an intent and gets back a **complete, self-contained artifact**. A **component** (`knowledge`, `preprocess`, `rag`, `library`, `mind`) is a building block a caller wires themselves.

Three rules keep the two altitudes honest:

- **A pipeline is pure processing: `input → artifact`.** It fetches, parses, chunks, structures, and produces any embeddings the artifact carries. It does **not** bind a store, persist, or retrieve. Producing the artifact *fully* is the whole job.
- **The artifact is a self-contained value.** It carries everything needed to use it — a structure tree's nodes hold their own text (and optionally an embedding), not a reference that must be refilled from a store later. Accept modest redundancy (a derived artifact copying some source text) to buy self-containment; do not trade it away to save bytes.
- **Persistence and retrieval are downstream.** `library` only dumps and loads that value (`put` / `open_*`); `mind` only retrieves. A caller chooses those steps; the pipeline does not do them. `library.put(x)` then `library.open_*(id)` must round-trip to an identical value.

**Retrieval returns values, not references.** A single-artifact retrieval reads content directly from the self-contained artifact and returns evidence with the content in it; a locator rides along only as optional provenance for cross-artifact fusion. If retrieval returned a bare reference, every call would be forced back through a store — the coupling this principle removes. Store-backed reference resolution is a corpus-scale concern (thousands of artifacts that do not fit in memory), not a single-artifact one.

**Half-finished intermediates are not pipelines.** "Parse only," "just the source revision," or "just the chunks" are component seams. Expose them from the component that owns them; do not promote them to a public flow just because a pipeline uses them internally.

**Smell test.** If a "pipeline" takes a `library` argument, writes to a store, or returns something you must resolve before you can read it, it has absorbed a downstream concern. Split the pure-processing part out and let the caller persist/retrieve.

## Callable Shape: Bind Config, Pass the Operand

Pick the shape by one question: **what must stay constant across repeated calls, and what varies?** Bind the constant at construction; pass only the varying operand per call.

- **Bind the config, not the operand.** An operation that will run over a batch binds its `cfg` at construction as immutable policy; each call passes only the runtime operand. `PaperFlow(cfg).build(input)` and `Retrieve(cfg).retrieve(tree, question)` bind `cfg` once so `batch_run(flow.build, inputs)` applies one unified setting to every input. This is a **reproducibility** requirement, not ergonomics: a config changed between or during calls silently yields results under mixed settings. Binding `cfg` is itself sufficient reason to be a class — an external dependency (a store, a provider) is *not* required, and its absence is *not* a reason to demote to a function.
- **Function** — fine for a genuinely one-off transform where nothing must stay constant across calls and passing `cfg` per call carries no reproducibility risk. If a batch would want one fixed setting, prefer a config-bound class instead.
- **The cfg *type* selects the behavior (typed dispatch).** One configured callable picks its shape or strategy from the **type** of the cfg it was built with: `PaperFlow(PaperStructureCfg).build(input)` builds tree-shaped knowledge; `Retrieve(AgenticRetrievalCfg).retrieve(...)` runs agentic traversal. cfg types may share a base cfg. This is internal typed dispatch and is explicitly **not** the forbidden retriever/flow class hierarchy or registry — the prohibition is on class trees, managers, and registries, not on one class branching by cfg type.
- **No mutable current-work state.** A config-bound service never stores a "current input" or "current result"; the operand goes in as an argument and the artifact comes out as a return value. A frozen `knowledge` artifact still never grows `build()`/`retrieve()`/persistence.

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
- [ ] Does the pipeline return a **self-contained** artifact (usable without a store), and does it avoid taking a `library`, persisting, or retrieving?
- [ ] Are persistence and retrieval left to the caller (`library` dump/load, `mind` retrieve), rather than folded into the pipeline?
- [ ] Does retrieval return evidence **values** (content included), with any locator as optional provenance — not a bare reference?
- [ ] Is `cfg` bound at construction (immutable) so a batch runs one unified setting, with only the operand passed per call?
- [ ] Does the callable pick its shape/strategy from the cfg **type** (typed dispatch), not from a class hierarchy or registry?
- [ ] Are you exposing a half-finished intermediate as a public flow? If so, keep it a component seam.
