# Component Development Workflow

How to implement or refactor code under `quantmind/`. Read this before
writing code; the architecture constraints in root `AGENTS.md` / `CLAUDE.md`
apply throughout.

## General Loop

1. **Find the nearest existing pattern.** Locate the closest analogous
   implementation in the target module and follow its structure, naming,
   and test layout. Consistency beats novelty.
2. **Check the dependency contract** for your module (table below) before
   adding any import. `lint-imports` enforces these; if your design needs a
   forbidden import, the design is wrong — restructure, do not work around
   the contract.
3. **Implement small and flat.** Use functions for self-contained stateless
   transformations. Use a small service class when operations share reusable
   dependencies, policy, or lifecycle; do not add a class only as a namespace,
   and do not store an implicit mutable "current input" or "current result".
   Use `Protocol` over ABC and avoid framework-style class hierarchies. No
   meaningless wrappers (a method must add logic, abstraction, or a side effect
   beyond the call it wraps). No premature abstractions — extract shared code
   when the second real caller appears, not before.
4. **Add the unit test and the example** (sections below).
5. **Update the public surface** if needed: package exports, the relevant
   design or guide, and the catalog in `docs/README.md`. Update the root
   README only when its overview or quick start changes.
6. **Verify**: run targeted tests while iterating, deterministic required
   verification before handoff, and every applicable live-network component
   smoke test for changed public-network integrations.

## Module Routing

### Dependency contracts (enforced by import-linter)

| Module | May import from `quantmind.*` |
|--------|-------------------------------|
| `quantmind/utils/` | nothing (leaf) |
| `quantmind/knowledge/` | nothing (leaf) |
| `quantmind/configs/` | `knowledge` only |
| `quantmind/preprocess/` | `utils` only |
| `quantmind/rag/` | `preprocess` only |
| `quantmind/flows/`, `quantmind/magic.py` | apex — may import all of the above |

### `quantmind/knowledge/` — data standard

- Pydantic models, `frozen=True`, `extra="forbid"`.
- Every `BaseKnowledge` subclass **must** require `as_of: datetime`
  (financial time-sensitivity is mandatory) and a typed `source: SourceRef`
  (no bare strings).
- Canonical models do not select retrieval text or store vectors. Put
  rebuildable text projections in `quantmind.library`.
- Pick one shape: `FlattenKnowledge` (atomic card), `TreeKnowledge`
  (hierarchical artifact), or `GraphKnowledge` (placeholder) for conventional
  knowledge. Source-first paper revisions and independently versioned paper
  artifacts use their dedicated frozen models instead of `BaseKnowledge`.

### `quantmind/configs/` — operation cfg + typed inputs

- Extend `BaseFlowCfg`; inputs are Pydantic models or discriminated unions.
- Never `Dict[str, Any]` in signatures — model it.

### `quantmind/preprocess/` — deterministic data prep

- Fetch / format / clean / time utilities: deterministic, no LLM calls.
- Return frozen dataclasses (`Fetched`, `RawPaper`, ...), not dicts.
- Surface the common path at the package root (`from quantmind.preprocess
  import fetch_arxiv`), keep explicit submodule paths working.

### `quantmind/rag/` — opinionated document RAG

- Use LlamaIndex for chunking, indexing, retrieval, and ranking; add only the
  source/page/provenance conversion that QuantMind owns.
- Import deterministic inputs from `quantmind.preprocess`; preprocessing must
  never import RAG.
- Keep LlamaIndex types private. Return frozen QuantMind evidence values.
- Do not add a public retriever, vector-store, provider, backend registry, or
  generic query-engine hierarchy.

### `quantmind/flows/` and `quantmind/magic.py` — apex layer

- Public operations are intent-oriented async callables. Prefer a function for
  one self-contained operation; use a small service class when its constructor
  binds dependencies or policy reused across calls. Service instances must not
  hide the active input or result as mutable state, and side effects stay
  explicit.
- Semantic operations use the OpenAI Agents SDK directly (`Agent`,
  `@function_tool`, `output_type=`); deterministic operations do not add an
  LLM. Never wrap `from agents import ...` in a facade.
- Fan-out goes through `batch_run` (bounded concurrency, error policy);
  `batch_run` rejects `memory=` at the signature layer by design.

### `quantmind/mind/` — cognitive layer

- Lands via the Agents SDK migration (tracking: #71). Backends implement
  the `Memory` Protocol with granular `tools()`, `mcp_servers()`,
  `run_hooks()`, `reset()` — each may return an empty list; do not force
  MCP on every implementation.

### `quantmind/utils/`

- Logger only. New general-purpose helpers need maintainer sign-off via an
  issue first; the default answer is "put it in the module that uses it".

## Public Operation Checklist

A public operation is complete only when all of these agree:

1. A stage and name consistent with `contexts/design/operations/naming.md`.
2. Typed input and config models, exported from `quantmind.configs`.
3. One intent-oriented async function or small service class exported from
   `quantmind.flows`, with its result contract exported from the canonical
   owning layer.
4. Offline success and failure tests for the public callable, plus a
   magic-introspection test when a function follows the `(input, *, cfg)`
   convention.
5. One runnable common-path example under `examples/<module>/`.
6. A relevant design or guide and one row in `docs/README.md`.

Do not add a registry solely for discovery; package exports and the component
catalog are the discovery surfaces.

## Public-Network Source Checklist

When adding a source to an existing operation:

1. Update the typed source selection and the operation's direct dispatch.
2. Keep acquisition policy internal. Add a shared provider abstraction only
   after a second implementation demonstrates shared behavior.
3. Add offline mocked tests for parsing, boundaries, continuation after item
   failures, and completeness semantics.
4. Update the source table and design under `docs/`.
5. Add or update a component-specific
   `scripts/verify_<component>_e2e.py` and list its command in
   `docs/README.md`.
6. Add or update the component's named job in the existing
   `.github/workflows/e2e.yml` and extend the workflow's precise pull-request
   path filter. When multiple live jobs exist, use GitHub-native per-job change
   detection so only affected component jobs run. Do not add a separate
   workflow, generic runner or registry, or base E2E class. Keep live network
   work out of `scripts/verify.sh`.

## Tests

- Location mirrors the module: `tests/<module>/test_<topic>.py`.
- Subclass `unittest.TestCase` (run via pytest).
- Mock external services (network, LLM APIs, filesystem where practical);
  tests must pass offline.
- Cover the success path **and** at least one failure path per public
  function.
- Coverage floor is enforced by `pytest --cov` in `scripts/verify.sh`; new
  code should not lower branch coverage.

## Examples

- One focused example per new feature under `examples/<module>/`
  (create the directory if it does not exist yet).
- Demo the simple, common usage — one scenario per file, runnable as
  `python examples/<module>/<name>.py`, minimal setup.

## Documentation

- Docstrings: English, Google style, required on public functions and
  models.
- Public behavior changes update the relevant design or guide and the catalog
  in `docs/README.md`. Update the root README only for top-level positioning
  or quick-start changes.
