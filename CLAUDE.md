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

## Current Repository State (transitional, after PR #70)

Surviving modules — these still work but will be replaced or migrated in PR2-PR4:

| Module | Status | Replacement |
|--------|--------|-------------|
| `quantmind/flow/` | active | `flows/` in PR4 |
| `quantmind/parsers/` | active | `preprocess/format/` in PR3 |
| `quantmind/sources/` | active | `preprocess/fetch/` in PR3 |
| `quantmind/config/` | active | `configs/` in PR2 |
| `quantmind/llm/` | active | deleted in PR4 (use SDK + `openai` directly) |
| `quantmind/models/{content,paper,analysis}.py` | active | move to `knowledge/` in PR2 |
| `quantmind/utils/logger.py` | active | permanent |

## Development Commands

### Environment

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Lint + Tests

```bash
ruff format .
ruff check .
pytest tests/
```

Pre-commit hooks (`.pre-commit-config.yaml`) run on push: trailing whitespace, EOF,
ruff, ruff-format, full pytest. Don't bypass hooks unless the user explicitly
authorizes — fix the underlying issue instead.

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

- **Schemas**: Pydantic, `extra="forbid"`, `frozen=True`. All `KnowledgeItem`
  subclasses must require `as_of: datetime` (financial time-sensitivity is mandatory)
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
| #70 (merged or in review) | Clean removal of self-built agent runtime |
| PR2 | `knowledge/` + `configs/` skeleton |
| PR3 | `preprocess/` (fetch + format two layers) |
| PR4 | `flows/` + `paper_flow` + `batch_run` + `magic.py`; drop old `flow/` `llm/` |
| PR5 | `mind/memory/filesystem` MVP + trajectory archive |
| PR6+ | Second flow (news/earnings) / observability cookbook / longer-term modules |
