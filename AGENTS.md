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
| `quantmind/rag/` | Opinionated LlamaIndex document chunking and retrieval — depends only on `preprocess` |
| `quantmind/flows/` | Apex layer: public library operations (`paper_flow`, `collect_news`, `batch_run`) |
| `quantmind/magic.py` | `resolve_magic_input`: natural language → `(input, cfg)` |
| `quantmind/mind/` | Pure-agentic reasoning layer — memory + agentic (reasoning-based) retrieval where an LLM decides; mechanical retrieval (similarity / BM25) lives in `rag` / `library` |
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

1. **Library, not framework** — use functions for self-contained stateless
   transformations and small service classes that bind, at construction, the
   immutable `cfg`/policy/dependency that must stay constant across calls; the
   runtime operand is passed per call. Binding `cfg` for batch reproducibility
   alone justifies a class (`PaperFlow(cfg).build(input)`,
   `AgenticRetriever(cfg).retrieve(structure, q)`), and the cfg *type* may select the
   shape/strategy (typed dispatch, not a class hierarchy). Keep canonical values
   free of runtime service state; use `Protocol` over ABC, with no
   framework-style class hierarchies, plugin registries, hook discovery, or CLI.
2. **RAG data plane, not framework** — use LlamaIndex directly inside
   `quantmind.rag`; keep upstream types private and do not add retriever,
   vector-store, provider, or backend registries.
3. **Do not rebuild the agent runtime** — use `openai-agents` directly; no
   QuantMind-side facades over `from agents import ...`.
4. **Schema models vs runtime evidence** — user/LLM inputs and configs use
   extra-forbid Pydantic models; knowledge adds `frozen=True`; deterministic
   fetch, preprocessing, and collection values use frozen dataclasses when
   they do not need validation or JSON Schema (`Fetched`, `NewsBatch`).
5. **Import boundaries are contracts** — `import-linter` (configured in
   `pyproject.toml`) pins the dependency graph; never work around a failing
   contract.
6. **Absolute imports** across module boundaries.
7. **No meaningless wrappers** — a method must add logic, abstraction, or a
   side effect beyond the call it wraps; otherwise inline it.
8. **Name public operations by intent** — follow
   `contexts/design/operations/naming.md`; use stage verbs, and reserve
   `pipeline` for deliberate multi-stage composition. `flow` as a verb or
   `*_flow` function name is banned; `Flow` as a noun on a document handle
   (`PaperFlow`) is allowed.
9. **Pipelines produce self-contained artifacts** — a `flows` pipeline is pure
   processing (`input → artifact`) and returns a value usable *and storable*
   without a store; it does not bind a `library`, persist, or retrieve.
   `library` only dumps and loads (`put(artifact)` / `open_*`, round-tripping to
   an identical value); `mind` only retrieves, returning evidence **values**
   (content included) with any locator as optional provenance. A self-contained
   artifact carries its own text (and any embeddings) plus the minimal
   provenance metadata (`as_of` + a light source ref) needed to persist and
   time-query it standalone — never a reference refilled from a store. Keep that
   provenance metadata out of the artifact's `id` / `content_hash` (identity
   stays reproducible); share it via a light provenance base, not full
   `BaseKnowledge`. Accept modest redundancy to keep artifacts self-contained.
   Half-finished intermediates stay component seams, not public flows. See
   `contexts/design/operations/orchestration.md`.

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
