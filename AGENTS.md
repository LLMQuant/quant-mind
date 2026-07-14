# QuantMind — Agent Instructions

Guidance for coding agents contributing to this repository. Keep this file
aligned with `CLAUDE.md` (same core rules); update both in the same change.

## What This Is

QuantMind is a knowledge extraction and retrieval library for quantitative
finance, built **on top of** the OpenAI Agents SDK. It is a domain library,
not an agent framework: runtime, tracing, tool scaffolding, and multi-agent
handoff all come from `openai-agents`.

## Module Map

| Module | Role |
|--------|------|
| `quantmind/knowledge/` | Pydantic data standard (`FlattenKnowledge` / `TreeKnowledge` / `GraphKnowledge`) — dependency leaf |
| `quantmind/configs/` | Operation cfg + typed input models or unions (`BaseFlowCfg`, `NewsWindow`, `PaperInput`) — depends only on `knowledge` |
| `quantmind/preprocess/` | Deterministic fetch / format / clean / time utilities — depends only on `utils` |
| `quantmind/flows/` | Apex layer: agent-facing operations (`paper_flow`, `collect_news`, `batch_run`) |
| `quantmind/magic.py` | `resolve_magic_input`: natural language → `(input, cfg)` |
| `quantmind/mind/` | Cognitive layer (memory protocol); landing via the Agents SDK migration (#71) |
| `quantmind/utils/` | Logger only — keep it that way |

The pre-migration agent runtime was removed and archived on the
`archive/agent-runtime-final` branch. Reference it for history; never
resurrect it into master.

## Setup and Verification

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
bash scripts/verify.sh              # deterministic offline golden gate
python scripts/verify_news_e2e.py   # live PR Newswire component gate
```

`scripts/verify.sh` runs five fast-fail steps (`ruff format --check`,
`ruff check`, `basedpyright`, `lint-imports`, `pytest --cov`) and must remain
network-free. Public-network integrations have separate bounded live gates;
run each applicable gate when changing that component and before publishing.
The current component catalog and commands live in `docs/README.md`. Do not
bypass pre-commit / pre-push hooks unless the user explicitly authorizes it —
fix the underlying issue instead.

## Architecture Constraints (stable)

1. **Library, not framework** — functions over classes, `Protocol` over ABC,
   no plugin registries, no hook discovery, no CLI.
2. **Do not rebuild the agent runtime** — use `openai-agents` directly; no
   QuantMind-side facades over `from agents import ...`.
3. **Schema models vs runtime evidence** — user/LLM inputs and configs use
   extra-forbid Pydantic models; knowledge adds `frozen=True`; deterministic
   fetch, preprocessing, and collection values use frozen dataclasses when
   they do not need validation or JSON Schema (`Fetched`, `NewsBatch`).
4. **Import boundaries are contracts** — `import-linter` (configured in
   `pyproject.toml`) pins the dependency graph; never work around a failing
   contract.
5. **Absolute imports** across module boundaries.
6. **No meaningless wrappers** — a method must add logic, abstraction, or a
   side effect beyond the call it wraps; otherwise inline it.

## Tests and Examples

A new feature ships with a unit test **and** a focused example:

- Tests: `tests/<module>/`, subclass `unittest.TestCase`, mock external
  services, cover success and failure paths.
- Examples: `examples/<module>/`, one simple usage per file.
- Public operations and sources: update the catalog in `docs/README.md` and
  follow the `quantmind-dev` component checklist.

## Communication

- Commit messages: English, Conventional Commits.
- PR titles, PR bodies, and issue bodies: English.
- Code comments and docstrings: English, Google style.

## Development Workflows

For commit, pull-request, or component-implementation tasks, load the
`quantmind-dev` skill and follow the matching reference:

- `.agents/skills/quantmind-dev/SKILL.md` (this toolchain)
- `.claude/skills/quantmind-dev/SKILL.md` (Claude Code)

The two copies are identical; when changing the skill, update both in the
same change.
