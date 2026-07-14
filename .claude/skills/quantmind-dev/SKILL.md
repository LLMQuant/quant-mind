---
name: quantmind-dev
description: Contributor workflow for the QuantMind codebase. Covers commit format, pull request format, and component development across quantmind/ modules (knowledge, configs, preprocess, flows, mind, utils) with tests, examples, and verification. Use when committing, opening a PR, or implementing/refactoring QuantMind code.
---

# QuantMind Dev

Development workflow for contributing to the QuantMind codebase.

## Start Here

1. Use the repository [`contexts/README.md`](../../../contexts/README.md) to
   select the development context for this contribution.
2. Read the repository root `AGENTS.md` or `CLAUDE.md` for the stable
   architecture constraints and the module map.
3. Read `docs/README.md` when the task adds, changes, or uses a public
   operation or public-network source.
4. When creating, updating, or triaging an issue or pull request, use the
   canonical [repository label guidance](../../../contexts/dev/labels.md).
5. Pick exactly one workflow reference below; do not load the others.

## Select Workflow

- Committing staged work → `references/commit.md`
- Opening or updating a pull request → `references/pull-request.md`
- Implementing or refactoring anything under `quantmind/` →
  `references/develop-components.md` (read it **before** writing code)

A feature task usually chains all three: develop → commit → pull request.

## Rules

- `bash scripts/verify.sh` is the deterministic offline golden gate. Run it
  before every push and before marking a PR ready.
- Public-network integrations have separate bounded live component gates.
  Run every applicable gate listed in `docs/README.md` when changing that
  component and before publishing.
- Never bypass pre-commit / pre-push hooks (`--no-verify`) unless the user
  explicitly authorizes it.
- New features ship with a unit test and a focused example (see
  `references/develop-components.md`).
- This skill is mirrored under `.agents/skills/quantmind-dev/` and
  `.claude/skills/quantmind-dev/`; the copies are identical. When changing
  the skill, update both in the same change.

## Boundaries

- This skill is for contributing to QuantMind itself. It does not cover
  using QuantMind as a library in your own project.
- Product decisions (new modules, new dependencies, API redesigns) need an
  issue and maintainer discussion first; do not encode them here.
