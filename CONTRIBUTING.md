# Contributing to QuantMind

Thank you for contributing to QuantMind! This guide covers environment
setup and the contribution process. The canonical development workflow
(commit format, PR format, component development) lives in the
`quantmind-dev` skill — `.claude/skills/quantmind-dev/SKILL.md` /
`.agents/skills/quantmind-dev/SKILL.md` — and the repository-wide rules
live in `AGENTS.md` / `CLAUDE.md`. If you develop with a coding agent, it
will pick these up automatically.

## 🚀 Quick Setup

1. **Fork and clone** the repository
2. **Set up environment**:

   ```bash
   uv venv && source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:

   ```bash
   ./scripts/pre-commit-setup.sh
   ```

## ✅ Verification

`scripts/verify.sh` is the deterministic required verification for every PR:

```bash
bash scripts/verify.sh
```

It runs five fast-fail steps: `ruff format --check`, `ruff check`,
`basedpyright`, `lint-imports`, `pytest --cov` (with a branch-coverage
floor configured in `pyproject.toml`). It stays network-free, and the required
`.github/workflows/ci.yml` workflow runs the same harness after file-hygiene
hooks.

Public-network integrations have separate bounded live component smoke tests.
`.github/workflows/e2e.yml` owns their scheduled, manual, and path-filtered
jobs. Run every applicable test when changing that component and before
publishing; the current commands are listed in
[`docs/README.md`](docs/README.md). For example, PR Newswire uses:

```bash
python scripts/verify_news_e2e.py
```

This command uses the real public network. External PR Newswire availability
must not block unrelated changes, so the E2E workflow only runs for relevant
pull-request paths and is not a required merge check.

**Hooks**: the pre-commit stage runs formatting/lint and file hygiene
checks; the pre-push stage runs the full `scripts/verify.sh`. If a hook
fails, fix the issue — don't bypass with `--no-verify`.

```bash
# Run all pre-commit hooks manually
pre-commit run --all-files

# Run targeted tests while iterating
pytest tests/<module>/
```

## 📝 Development Standards

- **Architecture**: QuantMind is a domain library on top of the OpenAI
  Agents SDK — functions over classes, `Protocol` over ABC, no CLI, no
  agent-runtime rebuilding. See `AGENTS.md` / `CLAUDE.md` for the stable
  constraints and the module map.
- **Style**: Google-style docstrings, English comments, 80-char lines
  (enforced by ruff).
- **Types**: Pydantic for inputs, configs, and knowledge schemas; frozen
  dataclasses for deterministic runtime evidence; comprehensive type hints
  (`basedpyright` runs in standard mode).
- **Dependency boundaries**: enforced by `import-linter`
  (`pyproject.toml`); don't work around a failing contract.
- **Tests**: required under `tests/<module>/` (mirror the module
  structure), `unittest.TestCase`, mock external services, cover success
  and failure paths.
- **Examples**: one focused example under `examples/<module>/` for each
  new feature.
- **Public operations and sources**: update the component catalog in
  `docs/README.md`; public-network sources also require a bounded live check.

## 🔄 Pull Request Process

1. **Create a feature branch** from `master`.
2. **Follow Conventional Commits**: `type(scope): description`, in English.
3. **Verify before submitting**: deterministic verification and every
   applicable live-network component smoke test must be green.
4. **Submit the PR** using the template — English body, reference the
   related issue, and state the verification you performed.
5. Keep PRs small and focused
   ([Google eng practices](https://google.github.io/eng-practices/review/developer/small-cls.html)).

For significant changes (new modules, new dependencies, API redesigns),
open an [issue](https://github.com/LLMQuant/quant-mind/issues) to discuss
first.

## ❓ Questions?

- Check existing [issues](https://github.com/LLMQuant/quant-mind/issues)
- Review architecture patterns in existing code
- See `AGENTS.md` / `CLAUDE.md` for repository-wide rules

Thank you for contributing! 🚀
