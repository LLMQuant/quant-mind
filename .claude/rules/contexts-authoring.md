---
paths:
  - "contexts/**/*.md"
---

# Contexts authoring standard

You are editing a page under `contexts/`. Follow the QuantMind contexts authoring standard. `tests/test_contexts.py` (run inside `scripts/verify.sh`) enforces the structural rules below, so a page that ignores them fails the build — getting them right up front avoids a red verify.

Load-bearing rules:

- Open every `contexts/**/*.md` with `## Quick Summary`, then `## Contents`, in that order, both within the first 80 lines.
- The `## Contents` links must exactly match the page's `##` section headings (GitHub-style anchors: lowercase, punctuation stripped, spaces and hyphens collapsed to one hyphen).
- Register the page in its index and keep index links resolving: a design page in `contexts/design/README.md`, a dev route in `contexts/dev/README.md`.
- Do not hard-wrap prose to a fixed width. Keep each paragraph and list item on one physical line; tables, fenced code, and mermaid keep their own line structure.

Canonical source (read it before a non-trivial contexts change): `.claude/skills/quantmind-dev/references/write-contexts.md`. Run `bash scripts/verify.sh` before pushing a contexts change.
