# Commit Workflow

How to commit work in the QuantMind repository.

## Message Format

English, [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <summary>
```

- **type**: `feat` | `fix` | `refactor` | `docs` | `test` | `chore`
- **scope**: optional but preferred — use the module or area the change
  touches, matching existing history (`feat(memory): ...`,
  `chore(verify): ...`, `docs(news): ...`). Omit the scope only for
  repo-wide changes (`feat: flows + magic apex layer`).
- **summary**: imperative, lower-case start, no trailing period.

Examples from history:

```text
feat: preprocess fetch+format module (PR4 of OpenAI Agents SDK migration) (#75)
chore(verify): add scripts/verify.sh + ruff/basedpyright/import-linter/pytest --cov harness (#73)
fix(test): Prevent creation of `test_data` directory during unit tests (#41)
```

## One Logical Change per Commit

Keep each commit focused on a single logical change. Split unrelated edits
(e.g. a bug fix discovered while building a feature) into separate commits.
Do not mix mechanical reformatting with behavior changes.

## Before Committing

1. Inspect exactly what you are about to commit:

   ```bash
   git status
   git diff --staged
   ```

   Confirm no unintended files (scratch scripts, local configs, large
   artifacts) are staged.

2. Run verification appropriate to the change:
   - during development: targeted tests, e.g. `pytest tests/<module>/`
   - before push / handoff: `bash scripts/verify.sh` (deterministic required
     verification)

3. Commit. Pre-commit hooks run ruff format/check and file hygiene checks;
   the pre-push hook runs the full `scripts/verify.sh`.

## Hooks

If a hook fails, fix the underlying issue and commit again. Never use
`git commit --no-verify` or otherwise bypass hooks unless the user has
explicitly authorized it for that specific commit.
