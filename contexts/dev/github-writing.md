# GitHub Writing Style

## Quick Summary

- **Purpose**: Define how raw Markdown should be formatted in QuantMind GitHub issues, pull requests, discussions, and comments.
- **Read when**: Drafting or editing any text submitted to GitHub.
- **Key rule**: Keep each paragraph, list item, checklist item, table row, and blockquote paragraph on one line in the raw Markdown.
- **Required check**: Read the raw body back after writing and remove formatter-introduced hard wrapping without changing meaning.

## Contents

- [Do Not Wrap Prose at a Fixed Width](#do-not-wrap-prose-at-a-fixed-width)
- [Write Workflow](#write-workflow)

This guide's workflow (Issue/PR templates, `--body-file`, the raw-body read-back) is specific to text submitted to GitHub. Its prose rule — do not hard-wrap at a fixed width — also governs contexts pages stored in the repository; see the contexts authoring standard.

## Do Not Wrap Prose at a Fixed Width

GitHub wraps text for the viewer. Do not insert line breaks in the raw body at a fixed width, including 80 columns.

- Keep each paragraph on one line in the raw Markdown.
- Keep each list or checklist item on one line.
- Keep each blockquote paragraph on one line after `>`.
- Keep each Markdown table row on one line.
- Preserve line structure inside fenced code blocks.
- Use blank lines, headings, lists, code fences, tables, or an intentional Markdown hard break only when they express real structure.

Repository formatters and code line-length settings do not apply to GitHub body prose. In particular, do not run Issue or Pull Request bodies through `textwrap.fill`, an 80-column editor wrap, or a Markdown formatter configured to wrap prose.

## Write Workflow

1. Draft the complete body with one paragraph or list item per source line.
2. Preserve the selected Issue or Pull Request template and its checklist.
3. Submit the body as written, preferably with `--body-file` when using `gh`.
4. Read the raw body back after creation or editing and check that prose was not hard-wrapped before considering the write complete.

When Prettier is already available, use its no-wrap mode to check the temporary body file before submission:

```bash
prettier --write --parser markdown --prose-wrap never /tmp/github-body.md
prettier --check --parser markdown --prose-wrap never /tmp/github-body.md
```

Do not introduce a repository-wide Markdown formatter or change Python's 80-column setting for this purpose. This check applies only to the temporary GitHub body.

When updating an existing hard-wrapped body, normalize the complete body while preserving its meaning, headings, lists, code blocks, links, and checkboxes.
