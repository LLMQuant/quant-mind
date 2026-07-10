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

`scripts/verify.sh` is the single source of truth for "is this branch
shippable". CI runs the exact same script, so a green local run means a
green PR:

```bash
bash scripts/verify.sh
```

It runs five fast-fail steps: `ruff format --check`, `ruff check`,
`basedpyright`, `lint-imports`, `pytest --cov` (with a branch-coverage
floor configured in `pyproject.toml`).

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
- **Types**: Pydantic models at boundaries, frozen dataclasses internally;
  comprehensive type hints (`basedpyright` runs in standard mode).
- **Dependency boundaries**: enforced by `import-linter`
  (`pyproject.toml`); don't work around a failing contract.
- **Tests**: required under `tests/<module>/` (mirror the module
  structure), `unittest.TestCase`, mock external services, cover success
  and failure paths.
- **Examples**: one focused example under `examples/<module>/` for each
  new feature.

## 🔄 Pull Request Process

1. **Create a feature branch** from `master`.
2. **Follow Conventional Commits**: `type(scope): description`, in English.
3. **Verify before submitting**: `bash scripts/verify.sh` must be green.
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
