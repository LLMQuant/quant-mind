# QuantMind Design

## Quick Summary

- **Purpose**: Find accepted designs for QuantMind packages and public
  operations.
- **Read when**: A task changes architecture, package responsibilities, public
  behavior, or a required validation rule.
- **Load next**: Open only the domain design that matches the task, then read
  that page in full before implementation.
- **Authority**: These pages explain agreed behavior. Each page lists any part
  that is not implemented yet.

## Contents

- [Design Index](#design-index)
- [Organization Rules](#organization-rules)

This directory records QuantMind engineering decisions. Use it to understand
which package owns each step, how packages work together, and what behavior an
implementation must preserve.

## Design Index

| Domain | Design |
|---|---|
| Flow | [Source-first paper flow](flow/paper.md) |
| Knowledge | [Paper sources, artifacts, citations, and locators](knowledge/paper.md) |
| Flow | [News collection](flow/news.md) |
| Preprocess | [Page-aware multimodal PDF parsing](preprocess/pdf.md) |
| RAG | [Page-aware document chunking and retrieval](rag/document.md) |
| Library | [Local knowledge storage and meaning-based search](library/local.md) |
| Mind | [Build and retrieve from a page-preserving structure tree](mind/retrieval.md) |
| Operations | [Public operation naming](operations/naming.md) |
| Operations | [Orchestration and construction altitude](operations/orchestration.md) |

## Organization Rules

- Organize designs directly by package or feature, such as `flow/`, `knowledge/`,
  `library/`, `operations/`, or `preprocess/`. Do not add an intermediate
  `components/` directory.
- Keep this page as the single global design index. Domain directories do not
  need their own index.
- Add a design page only for real design content. Do not create empty
  directories, placeholders, or speculative component pages.
- State whether a document describes current behavior, planned behavior, or
  both. List current gaps so readers can distinguish a plan from working code.
- Write headings and summaries as an action plus a concrete object. Avoid
  unexplained project shorthand and define any necessary domain term when it
  first appears.
- Use code and tests to check current behavior. Keep `docs/` focused on
  user-facing guides, examples, and catalogs; those pages may link here but
  must not maintain a second copy of a design.
