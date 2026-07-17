# GitHub Writing Style

## Quick Summary

- **Purpose**: Define the source-format contract for QuantMind GitHub Issue,
  Pull Request, Discussion, and comment bodies.
- **Read when**: Drafting or editing any text submitted to GitHub.
- **Key rule**: Keep each logical paragraph, list item, checklist item, table
  row, and blockquote paragraph on one physical source line.
- **Required check**: Read the raw body back after writing and remove
  formatter-introduced hard wrapping without changing meaning.

## Contents

- [No Hard-wrapped Prose](#no-hard-wrapped-prose)
- [Write Workflow](#write-workflow)

This guide is the canonical source-format contract for QuantMind Issue, Pull
Request, Discussion, and comment bodies. It applies to text submitted to
GitHub, not to Markdown files stored in the repository.

## No Hard-wrapped Prose

GitHub wraps rendered prose for the viewer. Do not insert source newlines at a
fixed width, including 80 columns.

- Keep each logical paragraph on one physical source line.
- Keep each list or checklist item on one physical source line.
- Keep each blockquote paragraph on one physical source line after `>`.
- Keep each Markdown table row on one physical source line.
- Preserve line structure inside fenced code blocks.
- Use blank lines, headings, lists, code fences, tables, or an intentional
  Markdown hard break only when they express real structure.

Repository formatters and code line-length settings do not apply to GitHub
body prose. In particular, do not run Issue or Pull Request bodies through
`textwrap.fill`, an 80-column editor wrap, or a Markdown formatter configured
to wrap prose.

## Write Workflow

1. Draft the complete body with one logical prose unit per source line.
2. Preserve the selected Issue or Pull Request template and its checklist.
3. Submit the body as written, preferably with `--body-file` when using `gh`.
4. Read the raw body back after creation or editing and check that prose was
   not hard-wrapped before considering the write complete.

When Prettier is already available, use its no-wrap mode as a preflight on the
temporary body file:

```bash
prettier --write --parser markdown --prose-wrap never /tmp/github-body.md
prettier --check --parser markdown --prose-wrap never /tmp/github-body.md
```

Do not introduce a repository-wide Markdown formatter or change Python's
80-column setting for this purpose. The preflight applies only to the temporary
GitHub body.

When updating an existing hard-wrapped body, normalize the complete body while
preserving its meaning, headings, lists, code blocks, links, and checkboxes.
