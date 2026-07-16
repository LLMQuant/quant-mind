# Public Operation Naming

## Scope

This document defines how public QuantMind callables communicate intent. It is
a naming contract, not a package-layout migration. Existing names may be
changed in focused compatibility work; they are not silently renamed here.

The runtime library API is designed for Python callers. Repository guidance
for coding agents belongs to the development harness (`AGENTS.md`, skills,
docs, fixtures, and verification), not in the runtime API description.

## Operation Stages

Public names use a stage verb plus the domain or result:

| Stage | Name pattern | Transformation |
|-------|--------------|----------------|
| Collection | `collect_<domain>` | External sources to source-faithful documents and evidence |
| Knowledge extraction | `extract_<domain>_knowledge` | Documents to typed `quantmind.knowledge` values |
| Index construction | `build_<domain>_index` | Documents or knowledge to a retrieval index |
| Analysis | `analyze_<domain>` | Domain inputs to an analytical result |
| Generic execution | `batch_run` and similar combinators | Apply an operation without changing its domain meaning |
| Composed recipe | `<domain>_<purpose>_pipeline` | Compose multiple named stages into a reusable pipeline |

`flow` is not a domain verb. Do not add new public `*_flow` names merely
because a callable performs several steps. Use a precise operation name, or a
`*_pipeline` name only when the callable deliberately composes multiple public
stages as a reusable recipe.

## Type Names

- Input types describe caller intent, such as `NewsWindow` or `PaperInput`.
- Config types name the domain and stage, such as `NewsCollectionCfg` or a
  future `PaperExtractionCfg`.
- Result types describe returned data, such as `NewsBatch`, `Paper`, or
  `PageIndex`.
- Provider names stay out of public operation names unless provider-specific
  behavior is itself the public contract.

## Current API

- `collect_news` is a collection operation and follows this contract.
- `batch_run` is a generic execution combinator, not a news or paper stage.
- `paper_flow` is an existing semantic-extraction API with legacy naming. It
  is not the naming precedent for new operations; any rename belongs in a
  separate compatibility change.

The current `quantmind.flows` package remains the apex implementation namespace
for this release. Whether it should become `operations` or be split from a
future `pipelines` package is deliberately outside this document.

## Review Checklist

Before adding a public callable, state:

1. Its operation stage.
2. Its input and result contracts.
3. Whether it is one operation or a genuine multi-stage pipeline.
4. Why its verb matches the observable result.

If those answers are unclear, settle the contract before adding a public name.
