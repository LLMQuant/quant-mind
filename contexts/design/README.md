# QuantMind Design

This directory is the canonical home for QuantMind engineering design. Use it
for accepted ownership boundaries, cross-domain contracts, and target behavior
that implementation work must preserve.

## Design Index

| Domain | Design |
|---|---|
| Flow | [Paper end-to-end contract](flow/paper.md) |
| Flow | [News collection](flow/news.md) |
| Library | [Local semantic knowledge library](library/local.md) |
| Operations | [Public operation naming](operations/naming.md) |

## Organization Rules

- Organize designs directly by owning domain, such as `flow/`, `knowledge/`,
  `library/`, `operations/`, or `preprocess/`. Do not add an intermediate
  `components/` directory.
- Keep this page as the single global design index. Domain directories do not
  need their own index.
- Add a design page only for real design content. Do not create empty
  directories, placeholders, or speculative component pages.
- State whether a document describes current behavior, a target contract, or
  both. Target contracts must identify current gaps so readers do not mistake
  an intended guarantee for an implemented one.
- Keep code and tests authoritative for current runtime behavior. Keep `docs/`
  focused on user-facing guides, examples, and catalogs; those pages may link
  here but must not maintain a second design copy.
