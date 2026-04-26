# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Project Overview

QuantMind is an intelligent knowledge extraction and retrieval framework for quantitative
finance. As of 2026-04, it is being **repositioned as a domain library that runs on top
of OpenAI Agents SDK**, rather than as a self-contained agent framework.

The pre-pivot agent runtime (`brain/`, `tools/`, `storage/`, `tagger/`, custom Tool ABC,
custom MultiStepAgent / Memory) was removed in PR #70. A full snapshot of the removed
code is preserved on the `archive/agent-runtime-final` branch on origin — reference it
if you need historical context, never resurrect it into master.

## Target Architecture (post-migration)

```
quantmind/
├── flows/        # e2e pipeline functions (paper_flow, news_flow, ...)
├── knowledge/    # Pydantic schemas (KnowledgeItem subclasses: Paper, News, ...)
├── preprocess/   # fetch (arxiv/http/doi/local) + format (pdf/html/markdown)
├── mind/         # cognitive layer; mind/memory/ is the MVP (filesystem-backed)
├── configs/      # centralized cfg + input types (BaseFlowCfg + per-flow types)
├── magic.py      # resolve_magic_input: natural language -> (input, cfg)
└── utils/        # logger only
```

Key principle: QuantMind does NOT rebuild Agent runtime, lifecycle hooks, tracing,
multi-agent handoff, or tool framework. Those come from `openai-agents`.

## Current Repository State (transitional, after PR #70 / #73 / PR3)

| Module | Status | Notes |
|--------|--------|-------|
| `quantmind/knowledge/` | landed (PR3) | data standard with three shapes: `FlattenKnowledge` (`News` / `Earnings` / `PaperKnowledgeCard`), `TreeKnowledge` (`Paper`), `GraphKnowledge` (placeholder); shared base = `BaseKnowledge` with typed `SourceRef` / `ExtractionRef` provenance + `embedding_text()` contract |
| `quantmind/configs/` | landed (PR3) | `BaseFlowCfg` / `BaseInput` + per-flow cfg + discriminated-union input types |
| `quantmind/utils/logger.py` | permanent | only general-purpose utility |
| `quantmind/flow/` | transitional | replaced by `flows/` in PR5 |
| `quantmind/parsers/` | transitional | replaced by `preprocess/format/` in PR4 |
| `quantmind/sources/` | transitional | replaced by `preprocess/fetch/` in PR4 |
| `quantmind/config/` | transitional | superseded by `quantmind/configs/` (PR3); deletion when consumers (`flow/`, `llm/`) migrate in PR5 |
| `quantmind/llm/` | transitional | deleted in PR5 (use SDK + `openai` directly) |
| `quantmind/models/{content,paper,analysis}.py` | transitional | superseded by `quantmind/knowledge/` (PR3); deletion when consumers (`parsers/`, `sources/`, `flow/`) migrate in PR4-PR5 |
| `quantmind/utils/tmp.py` | transitional | deleted in PR3-4 alongside the templating refactor |

Transitional modules are excluded from `basedpyright` (see `pyproject.toml`)
to keep the harness green during migration. New modules (`knowledge/`,
`configs/`, `preprocess/`, `flows/`, `mind/`, `magic.py`) are auto-included
at standard mode and gated by additional `import-linter` contracts so they
cannot accidentally pull in a transitional module.

## Development Commands

### Environment

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Verify (canonical local check)

`scripts/verify.sh` is the single source of truth for "is this branch
shippable". CI (`.github/workflows/verify.yml`) runs the exact same script,
so a green local run means a green PR. Run it before every push:

```bash
bash scripts/verify.sh
```

It runs five steps in fixed order, fast-failing on the first error:

1. `ruff format --check` — formatting must be clean
2. `ruff check` — lint (D, E, F, I, W, B, W505) must pass
3. `basedpyright` — standard-mode type check on permanent + new modules
4. `lint-imports` — architectural boundary contracts must hold
5. `pytest --cov` — tests pass with ≥ 60% branch coverage (will ratchet up
   to 70%+ in PR3 once transitional modules are removed)

Pre-commit hooks (`.pre-commit-config.yaml`):
- pre-commit stage: trailing whitespace / EOF / ruff / ruff-format (fast)
- pre-push stage: full `scripts/verify.sh`

Don't bypass hooks unless the user explicitly authorizes — fix the underlying
issue instead.

## Architecture Principles

1. **No framework, just lib** — Functions over classes; Protocol over ABC; no plugin
   registries or hook discovery
2. **Pure functions** — Flows are `async def run(...)`, not classes; state passed as
   args; side effects via explicit hooks
3. **Pydantic at boundaries, frozen dataclass internally** — Pydantic for anything
   exposed to LLM (`output_type=`, cfg, input); frozen dataclass for internal value
   types
4. **Batch is first-class** — `batch_run(flow_fn, inputs, ...)` will land in PR4
   (concurrency + error handling + progress aggregation). Users do NOT write
   `asyncio.gather` boilerplate themselves
5. **Customization 3 layers** — cfg (YAML/CLI), kwargs (Python `extra_*` flow args),
   building blocks (fork the flow file). Each layer has explicit extension points
6. **Observability 3 layers** — SDK auto-tracing, external processors via
   `add_trace_processor()`, local trajectory archive under `<memory_dir>/runs/`
7. **No CLI** — User-facing entry is a runbook script (5 lines of Python), not a
   framework command. Magic input is the loose-input UX, resolved by an Agent
8. **Magic input first** — Users describe intent in natural language;
   `magic.resolve_magic_input(...)` returns a structured `(input, cfg)` tuple

## Conventions When Editing

- **Schemas**: Pydantic, `extra="forbid"`, `frozen=True`. All `BaseKnowledge`
  subclasses must require `as_of: datetime` (financial time-sensitivity is mandatory)
  and provide a typed `source: SourceRef` (no bare strings). Subclasses MUST
  override `embedding_text()` so the store layer knows what to embed.
- **Knowledge shapes**: pick one of `FlattenKnowledge` (atomic card),
  `TreeKnowledge` (hierarchical artifact), or wait for `GraphKnowledge`
  (placeholder). Whole-document objects are `TreeKnowledge` even when a
  flatten card exists alongside (e.g. `Paper` vs `PaperKnowledgeCard`).
- **Configs**: Extend `BaseFlowCfg` (lands in PR2); never use `Dict[str, Any]` in
  init signatures
- **Tools**: SDK's `@function_tool` decorator; do NOT subclass anything
- **Memory backends**: Implement the `Memory` Protocol with granular `tools()`,
  `mcp_servers()`, `run_hooks()`, `reset()` — each may return an empty list. Do not
  force MCP on every implementation
- **Tests**: Subclasses of `unittest.TestCase` in `tests/<module>/`. Mock external
  dependencies; cover both success and failure paths
- **Imports**: Absolute (`from quantmind.knowledge import Paper`); no relative
  imports across module boundaries

## Communication Conventions

- **PR descriptions and issue bodies must be written in English**, regardless of the
  language of the conversation that triggered them. They are read by external audiences
  (search indexers, future maintainers, contributors who don't read Chinese).
- Commit messages: English, conventional-commit style (`feat:` / `fix:` / `refactor:` /
  `docs:` / `chore:` ...).
- Inline PR review comments and issue discussion threads may be in whichever language
  fits the participants.

## Things NOT to Do

- ❌ Rebuild Agent runtime / Tool ABC / lifecycle hook abstraction
- ❌ Add a CLI (`argparse`/`typer`/`click`); users run Python runbook scripts
- ❌ Introduce class-based `BaseFlow` / plugin registry / hook discovery
- ❌ Wrap `from agents import ...` in a QuantMind-side facade — use the SDK directly
- ❌ Mix `batch_run` and `memory` (they will be mutually exclusive in MVP; see PR5)
- ❌ Use `Dict[str, Any]` in init functions; use Pydantic models
- ❌ Add hard deps on observability platforms (Langfuse / Logfire / etc.); document
  integration via `add_trace_processor()` in user-facing cookbook only
- ❌ Build embedding-based memory before filesystem memory has shipped and stabilized

## Reference Material

- OpenAI Agents SDK docs: <https://openai.github.io/openai-agents-python/>
- Lifecycle / RunHooks API: <https://openai.github.io/openai-agents-python/ref/lifecycle/>
- MCP integration (filesystem server): <https://openai.github.io/openai-agents-python/mcp/>
- Tracing (auto-capture, processors, disable): <https://openai.github.io/openai-agents-python/tracing/>
- Original SDK announcement: <https://openai.com/index/the-next-evolution-of-the-agents-sdk/>
- Removed agent runtime snapshot: `archive/agent-runtime-final` branch on origin

## Roadmap (post-PR1)

| PR | Focus |
|----|-------|
| #70 (merged) | Clean removal of self-built agent runtime |
| #73 (merged) | Golden Harness — `scripts/verify.sh` with ruff + basedpyright + import-linter + pytest --cov, plus matching CI |
| PR3 (this PR) | `knowledge/` data standard (Flatten / Tree / Graph shapes) + `configs/` skeleton; `openai-agents>=0.14` introduced for `BaseFlowCfg.model_settings`. Storage layer (`mind/store/` + SQLite + `sqlite-vec`) lands separately. |
| PR4 | `preprocess/` (fetch + format two layers); migrate `parsers/` + `sources/`; delete `quantmind/models/{content,paper,analysis}.py` |
| PR5 | `flows/` + `paper_flow` + `batch_run` + `magic.py`; delete `quantmind/flow/`, `quantmind/llm/`, `quantmind/config/` |
| PR6 | `mind/memory/filesystem` MVP + trajectory archive |
| PR7+ | Second flow (news/earnings) / observability cookbook / longer-term modules |
