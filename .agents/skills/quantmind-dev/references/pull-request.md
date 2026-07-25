# Pull Request Workflow

How to open and maintain a pull request against QuantMind.

## Before Opening

1. Branch from `master`; keep the PR small and focused — one PR equals one
   reviewable change (see
   [Google eng practices on small CLs](https://google.github.io/eng-practices/review/developer/small-cls.html)).
2. Run the deterministic required verification:

   ```bash
   bash scripts/verify.sh
   ```

3. If the change affects a public-network integration, run every applicable
   live-network component smoke test listed in `docs/README.md`.

Do not open or mark ready a PR with red deterministic verification or an
applicable red live-network smoke test.

## Title

English, Conventional Commit style, same format as commit messages:

```text
<type>(<scope>): <summary>
```

Examples: `feat(preprocess): add RSS fetcher`, `fix(flows): handle empty
batch input`, `docs(readme): update quick start`.

## Body

Written in English (external audiences read it: contributors, search
indexers, future maintainers). Follow `.github/PULL_REQUEST_TEMPLATE.md`
and the canonical
GitHub writing style in `contexts/dev/github-writing.md`. Do not
hard-wrap body prose at 80 columns or another fixed width. Make sure the body
covers:

1. **What changed and why** — a short summary; link the design discussion
   if one exists.
2. **Related issue** — reference it when one exists (`Closes #NN` /
   `Part of #NN`).
3. **Verification performed** — state the exact deterministic and applicable
   live-network commands you ran (e.g. `bash scripts/verify.sh`; targeted
   `pytest tests/<module>/`; a component-specific verifier; manual example
   run) so the reviewer does not have to guess.

Keep the template checklist and remove items that do not apply.

## Review and Merge

- Assign relevant reviewers when you know who owns the area.
- Address review comments with follow-up commits (no force-push rewrites of
  reviewed history unless asked).
- After merge: switch back to `master`, pull, and delete the feature branch.
