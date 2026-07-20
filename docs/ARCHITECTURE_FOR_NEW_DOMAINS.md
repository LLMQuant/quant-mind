# Extending QuantMind Beyond a Single Domain

QuantMind currently ships a paper-oriented extraction flow, but the architecture
is meant to support broader agentic knowledge work. This guide explains which
parts are stable, which parts are domain-specific today, and how to extend the
stack without fighting the current design.

## What is stable today

The permanent architecture is:

```text
quantmind/
├── flows/        # apex orchestration layer
├── knowledge/    # typed knowledge outputs
├── preprocess/   # fetch + format + clean
├── configs/      # typed flow inputs and cfg
├── mind/         # memory/store roadmap
├── magic.py      # natural-language resolver
└── utils/        # logger
```

These layers already work well for agent teams because they create a strict
contract between source material, extraction, and downstream reuse.

## What is domain-specific today

The first production flow is `paper_flow`, and its output schema is
`quantmind.knowledge.Paper`. That flow is optimized for research-paper style
documents and still includes some finance-origin fields such as `asset_classes`.

That does **not** make the whole repository finance-only. It means the current
reference implementation is paper-first while the underlying architecture is
already reusable:

- `preprocess/` is domain-agnostic
- `magic.resolve_magic_input()` is flow-agnostic
- `BaseFlowCfg` and `BaseInput` are reusable contracts
- `BaseKnowledge` and the three knowledge shapes are reusable patterns
- `batch_run()` is reusable for stateless fan-out work

## How to add a new domain

### 1. Pick the right knowledge shape

Use:

- `FlattenKnowledge` for atomic records, events, or summaries
- `TreeKnowledge` for long documents with section/subsection structure
- `GraphKnowledge` only when the relationship layer is actually ready to land

Start from the retrieval shape you want downstream agents to use, not from the
source format you happen to ingest first.

### 2. Define typed inputs and configuration

Add a new config module under `quantmind/configs/`:

- define one or more `BaseInput` subclasses
- create a discriminated union for the flow input
- extend `BaseFlowCfg` with only domain-relevant knobs

The goal is for agents and humans to share the same explicit input contract.

### 3. Implement a pure flow function

Add a new `async def ..._flow(...)` under `quantmind/flows/` that:

- accepts one typed input plus `cfg=...`
- fetches or normalizes source content through `preprocess/`
- runs an Agents SDK `Agent(output_type=...)`
- returns a typed knowledge object

Do not introduce a custom runtime, plugin registry, or class-based flow
hierarchy.

### 4. Make the flow usable from natural language

If the flow follows the same `(input, *, cfg, ...)` signature convention,
`magic.resolve_magic_input()` can already resolve free-form intent into typed
input and config objects.

This is the bridge that helps external agent systems work with QuantMind
without writing brittle prompt parsers.

## Recommended agentic loop patterns

### Stateless scout loop

Use when you want breadth first:

1. resolve or build many typed inputs
2. run the flow in parallel with `batch_run()`
3. collect typed outputs
4. hand those outputs to a review or ranking agent

This is the best fit for discovery, triage, and broad corpus scanning.

### Serial memory loop

Use when each step depends on previous results:

1. read one document
2. extract a typed knowledge object
3. update memory or a local run archive
4. feed the next item with that accumulated context

Today, this pattern should be implemented as an explicit serial loop. `batch_run`
intentionally rejects `memory=` because shared-memory fan-out is not yet the
MVP design.

### Multi-agent handoff loop

A simple, durable pattern is:

1. **Scout agent** resolves intent and gathers sources
2. **Extractor agent** creates typed knowledge objects
3. **Reviewer agent** validates structure, provenance, and confidence
4. **Planner agent** decides what to fetch or revisit next

The key rule is to pass typed artifacts between agents whenever possible. Avoid
hidden agreements in prompt text when a Pydantic object can carry the contract.

## Memory and durability roadmap

The repository is explicitly moving toward a memory-aware architecture:

- `mind/memory.py` provides the current L1/L2/L3 memory primitives
- `mind/store/` remains the planned retrieval/store layer
- `flows/_runner.py` already reserves the runtime seam for memory integration
- `flows/governance.py` enforces policy for loop budgets, tool allowlists,
  fallback behavior, and L3 commit gates

Until those land, treat QuantMind as a strong typed-extraction layer with a
clear upgrade path toward durable agent loops.

## Documentation discipline

QuantMind is increasingly meant to be read by both humans and AI agents.
Because of that:

- keep `README.md`, `CONTRIBUTING.md`, and this file aligned with code changes
- document new domain assumptions in the same PR that introduces them
- review agent-facing docs regularly during active iteration

This discipline matters because stale documentation causes broken agent loops
just as quickly as stale code.
