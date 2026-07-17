# QuantMind — Agent Instructions

Guidance for coding agents contributing to this repository. Keep this file
aligned with `CLAUDE.md` (same core rules); update both in the same change.

Use [`contexts/README.md`](contexts/README.md) as the repository information
entry point for either development or library-usage work.

## Progressive Context Loading

Pages under `contexts/` are agent-facing references designed for progressive
disclosure:

1. Read lines 1-80 first. The preview contains `Quick Summary` and `Contents`
   sections that explain the page's purpose, authority, and scope.
2. Use that preview to decide whether the page applies. Do not preload sibling
   pages or follow unrelated links.
3. When a page applies, read the entire page before changing code, contracts,
   or repository guidance. The preview routes work; it does not replace the
   detailed contract.
4. Follow directly linked canonical sources only as the task requires. Avoid
   deep reference chains and duplicate guidance in working context.

## What This Is

QuantMind is a knowledge extraction and retrieval library for quantitative
finance, built **on top of** the OpenAI Agents SDK. It is a domain library,
not an agent framework: runtime, tracing, tool scaffolding, and multi-agent
handoff all come from `openai-agents`.

## Module Map

| Module | Role |
|--------|------|
| `quantmind/knowledge/` | Pydantic data standard (`FlattenKnowledge` / `TreeKnowledge` / `GraphKnowledge`) — dependency leaf |
| `quantmind/library/` | Local persistence and semantic retrieval for canonical knowledge — depends only on `knowledge` |
| `quantmind/configs/` | Operation cfg + typed input models or unions (`BaseFlowCfg`, `NewsWindow`, `PaperInput`) — depends only on `knowledge` |
| `quantmind/preprocess/` | Deterministic fetch / format / clean / time utilities — depends only on `utils` |
| `quantmind/flows/` | Apex layer: public library operations (`paper_flow`, `collect_news`, `batch_run`) |
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
bash scripts/verify.sh              # deterministic required verification
```

`scripts/verify.sh` runs five fast-fail steps (`ruff format --check`,
`ruff check`, `basedpyright`, `lint-imports`, `pytest --cov`) and must remain
network-free. `.github/workflows/ci.yml` is the required deterministic CI
workflow. Public-network integrations have separate bounded smoke tests;
`.github/workflows/e2e.yml` owns their scheduled, manual, and path-filtered
component jobs. Run each applicable smoke test when changing that component
and before publishing. External service availability must not block changes
outside that component. `docs/README.md` is the single catalog of component
commands; do not enumerate them in this file.

To extend live verification, add a component-specific
`scripts/verify_<component>_e2e.py`, add a named job to the existing `e2e.yml`,
extend its precise PR path filter, and add one catalog row. When a second live
job is added, use GitHub-native per-job change detection so PRs run only the
affected component jobs. Do not add another E2E workflow, a generic runner or
registry, or a base E2E class. Do not bypass pre-commit / pre-push hooks unless
the user explicitly authorizes it — fix the underlying issue instead.

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
7. **Name public operations by intent** — follow
   `contexts/design/operations/naming.md`; use stage verbs, and reserve
   `pipeline` for deliberate multi-stage composition.

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
- GitHub Issue, Pull Request, Discussion, and comment body formatting follows
  the [GitHub writing style](contexts/dev/github-writing.md); never hard-wrap
  remote prose at 80 columns or another fixed width.
- Code comments and docstrings: English, Google style.

## Development Workflows

For commit, pull-request, or component-implementation tasks, load the
`quantmind-dev` skill and follow the matching reference:

- `.agents/skills/quantmind-dev/SKILL.md` (this toolchain)
- `.claude/skills/quantmind-dev/SKILL.md` (Claude Code)

The two copies are identical; when changing the skill, update both in the
same change.
