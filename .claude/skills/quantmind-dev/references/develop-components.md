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
3. **Implement small and flat.** Pure functions over classes; `Protocol`
   over ABC; no meaningless wrappers (a method must add logic, abstraction,
   or a side effect beyond the call it wraps). No premature abstractions —
   extract shared code when the second real caller appears, not before.
4. **Add the unit test and the example** (sections below).
5. **Update the public surface** if needed: module `__init__.py` exports
   and user-facing docs (`README.md`, `docs/`) for user-visible features.
6. **Verify**: targeted `pytest tests/<module>/` while iterating;
   `bash scripts/verify.sh` before handoff.

## Module Routing

### Dependency contracts (enforced by import-linter)

| Module | May import from `quantmind.*` |
|--------|-------------------------------|
| `quantmind/utils/` | nothing (leaf) |
| `quantmind/knowledge/` | nothing (leaf) |
| `quantmind/configs/` | `knowledge` only |
| `quantmind/preprocess/` | `utils` only |
| `quantmind/flows/`, `quantmind/magic.py` | apex — may import all of the above |

### `quantmind/knowledge/` — data standard

- Pydantic models, `frozen=True`, `extra="forbid"`.
- Every `BaseKnowledge` subclass **must** require `as_of: datetime`
  (financial time-sensitivity is mandatory) and a typed `source: SourceRef`
  (no bare strings), and **must** override `embedding_text()`.
- Pick one shape: `FlattenKnowledge` (atomic card), `TreeKnowledge`
  (hierarchical artifact), or `GraphKnowledge` (placeholder). Whole-document
  objects are `TreeKnowledge` even when a flatten card exists alongside
  (e.g. `Paper` vs `PaperKnowledgeCard`).

### `quantmind/configs/` — flow cfg + typed inputs

- Extend `BaseFlowCfg`; inputs are discriminated-union Pydantic types.
- Never `Dict[str, Any]` in signatures — model it.

### `quantmind/preprocess/` — deterministic data prep

- Fetch / format / clean / time utilities: deterministic, no LLM calls.
- Return frozen dataclasses (`Fetched`, `RawPaper`, ...), not dicts.
- Surface the common path at the package root (`from quantmind.preprocess
  import fetch_arxiv`), keep explicit submodule paths working.

### `quantmind/flows/` and `quantmind/magic.py` — apex layer

- Flows are pure `async def` functions, not classes; state passes as
  arguments; side effects go through explicit hooks.
- Use the OpenAI Agents SDK directly (`Agent`, `@function_tool`,
  `output_type=`); never wrap `from agents import ...` in a facade.
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
- User-visible behavior changes update `README.md` (usage section) and
  `docs/` where applicable.
