# Contributor Setup and Issue Filing

How to set a working copy up for contribution, and how to file an issue.
Human-readable equivalent: root `CONTRIBUTING.md` (same steps, same order).

## Set Up as a Contributor

1. Get a working copy. External contributors fork on GitHub and clone the
   fork; maintainers branch directly off `master`:

   ```bash
   git clone https://github.com/<you>/quant-mind.git && cd quant-mind
   git checkout -b <type>/<short-topic>   # e.g. feat/sec-collection
   ```

2. Create the environment with [uv](https://github.com/astral-sh/uv):

   ```bash
   uv venv && source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

3. Install the git hooks (pre-commit lint/format stage plus the pre-push
   verify stage):

   ```bash
   ./scripts/pre-commit-setup.sh
   ```

4. Confirm the checkout is green before changing anything:

   ```bash
   bash scripts/verify.sh
   ```

   Verify is the single deterministic gate — format, lint, types, import
   boundaries, tests with coverage. CI runs the exact same script, so a green
   local run means a green PR. If it fails on a fresh checkout, stop and report
   that instead of working around it.

5. Continue with the workflow references: `develop-components.md` before
   writing code, then `commit.md` and `pull-request.md`.

## File an Issue

1. Requires an authenticated GitHub CLI (`gh auth status`; if missing, ask the
   user to run `gh auth login` themselves).
2. Pick the shape that matches the request:
   - Defect in existing behavior → `.github/ISSUE_TEMPLATE/bug_report.md`
   - Missing source, knowledge type, flow, or capability →
     `.github/ISSUE_TEMPLATE/feature_request.md`
3. Write the body in English following `contexts/dev/github-writing.md`
   (no hard-wrapping), and apply labels per `contexts/dev/labels.md`.
4. Search for duplicates first (`gh issue list --search "<keywords>"`), then:

   ```bash
   gh issue create --title "<imperative summary>" --body-file <tmpfile> \
     --label "<label>"
   ```

5. If the need came out of work in progress, link the issue from the related
   branch or PR so the discussion has code context.

## Boundaries

- Setup here targets contributing to QuantMind itself. Library-only users can
  stop after `uv pip install -e .` and do not need hooks.
- Do not file issues that encode product decisions as settled; state the need
  and let maintainer discussion pick the design (see SKILL.md Boundaries).
