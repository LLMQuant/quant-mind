# Contributing to QuantMind

Thank you for contributing to QuantMind.

QuantMind is now a **domain library on top of the OpenAI Agents SDK**, not a
self-contained agent framework. The first production flow is finance-first, but
the architecture is intentionally useful for broader agentic knowledge work.

## 🚀 Quick setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
./scripts/pre-commit-setup.sh
```

If `uv` is unavailable in your environment, a standard Python venv is an
acceptable fallback:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## ✅ Canonical verification loop

Run this before every push:

```bash
bash scripts/verify.sh
```

This is the single source of truth for branch health. It runs:

1. `ruff format --check`
2. `ruff check`
3. `basedpyright`
4. `lint-imports`
5. `pytest --cov`

CI runs the same script.

## 🏗️ Current architecture

The permanent module roots are:

- `quantmind/flows/` — apex orchestration layer
- `quantmind/configs/` — typed inputs and config
- `quantmind/knowledge/` — typed knowledge shapes and provenance
- `quantmind/preprocess/` — fetch + format + clean
- `quantmind/magic.py` — natural-language resolver
- `quantmind/mind/` — upcoming memory/store work
- `quantmind/utils/` — logger only

Do **not** re-introduce the deleted transitional/runtime packages such as
`quantmind.flow`, `quantmind.config`, `quantmind.llm`, or `quantmind.models`.

## 🧭 Design rules

- Prefer **pure functions** over framework classes.
- Use the **OpenAI Agents SDK directly**; do not build a new runtime wrapper.
- Keep **Pydantic models at boundaries** and small internal value types simple.
- Use **absolute imports** across module boundaries.
- Keep **comments and docstrings in English** with Google-style formatting.
- Preserve architectural boundaries enforced by `import-linter`.

## 🧪 Testing expectations

- Add or update tests under `tests/` when behavior changes.
- New feature work should cover both success and failure paths.
- Mock external services and network calls.
- Keep coverage above the configured floor.

For implementation work, add a simple example when it helps demonstrate the new
behavior clearly.

## 🧩 Common contribution paths

### New knowledge type

- Add a schema under `quantmind/knowledge/`
- Choose the right base shape (`FlattenKnowledge`, `TreeKnowledge`, or future
  `GraphKnowledge`)
- Implement `embedding_text()`
- Add tests under `tests/knowledge/`

### New flow

- Add a typed input/config module under `quantmind/configs/`
- Add a pure `async def ..._flow(...)` under `quantmind/flows/`
- Use `preprocess/` helpers instead of duplicating fetch/format logic
- Return a typed knowledge object
- Add tests under `tests/flows/`

### New source or format support

- Add leaf functionality under `quantmind/preprocess/`
- Keep `preprocess/` independent of `configs/`, `knowledge/`, and `flows/`
- Add focused tests under `tests/preprocess/`

### New domain extension

If you are adapting QuantMind beyond finance, read
`docs/ARCHITECTURE_FOR_NEW_DOMAINS.md` first and keep the extension aligned with
the existing layering.

## 🔄 Pull request expectations

Before submitting a PR:

```bash
pre-commit run --all-files
bash scripts/verify.sh
```

Also make sure:

- commits use conventional-commit style (`feat:`, `fix:`, `docs:`, `refactor:`)
- PR titles and descriptions are written in English
- docs are updated when behavior or extension points change

## 🤖 Agent-facing documentation

QuantMind is increasingly consumed by AI agents as well as humans. Treat the
following files as product surfaces:

- `README.md`
- `CONTRIBUTING.md`
- `docs/ARCHITECTURE_FOR_NEW_DOMAINS.md`

If you change architecture, memory semantics, or extension paths, update the
relevant documentation in the same PR. During active iteration, these agent-
facing docs should be reviewed regularly so agent workflows do not drift away
from the real codebase.

## ❓Questions?

- Check existing [issues](https://github.com/LLMQuant/quant-mind/issues)
- Review `CLAUDE.md` for the detailed architecture context
- Read `docs/ARCHITECTURE_FOR_NEW_DOMAINS.md` for extension guidance

Thank you for contributing.
