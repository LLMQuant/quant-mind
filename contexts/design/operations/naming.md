# Public Operation Names

## Quick Summary

- **Purpose**: Define clear names for public QuantMind functions and their input, config, and result types.
- **Read when**: Adding or renaming a public function, operation type, or pipeline.
- **Status**: Use these rules for new names. Keep the existing `paper_flow` name until a separate change explains how callers will migrate.
- **Core rule**: Use a verb that says what the function does. Use `pipeline` only when one function combines several public operations. Do not use `flow` as a verb.

## Contents

- [Scope](#scope)
- [Name Patterns](#name-patterns)
- [Type Names](#type-names)
- [Current API](#current-api)
- [Review Checklist](#review-checklist)

## Scope

This page defines how to name public QuantMind functions. It does not rename
packages or existing functions. Any existing name change needs a separate plan
for users who depend on the old name.

The runtime library API is designed for Python callers. Repository guidance
for coding agents belongs in `AGENTS.md`, skills, docs, fixtures, and checks;
it does not belong in the runtime API.

## Name Patterns

Start a public function name with a verb that describes its result:

| Work | Name pattern | What it does |
|-------|--------------|----------------|
| Collection | `collect_<domain>` | Fetch source documents and the details needed to verify them |
| Knowledge extraction | `extract_<domain>_knowledge` | Turn documents into typed `quantmind.knowledge` values |
| Index construction | `build_<domain>_index` | Turn documents or knowledge into a search index |
| Analysis | `analyze_<domain>` | Domain inputs to an analytical result |
| Generic execution | `batch_run` and similar helpers | Apply an operation to a batch without changing what the operation means |
| Pipeline | `<domain>_<purpose>_pipeline` | Combine several named operations into one reusable function |

`flow` is not a verb that explains a result. Do not add a public `*_flow` name
only because a function performs several steps. Name the result precisely, or
use `*_pipeline` when the function combines multiple public operations.

## Type Names

- Input types describe what the caller supplies, such as `NewsWindow` or
  `PaperInput`.
- Config types name the domain and stage, such as `NewsCollectionCfg` or a
  future `PaperExtractionCfg`.
- Result types describe returned data, such as `NewsBatch`, `Paper`, or
  `PageIndex`.
- Keep provider names out of public function names unless callers are choosing
  provider-specific behavior.

## Current API

- `collect_news` is a collection operation and follows these naming rules.
- `batch_run` is a generic batch helper, not a news or paper operation.
- `paper_flow` is an existing extraction API with an old name. Do not copy that
  pattern for new operations. Renaming it requires a separate migration plan.

The current `quantmind.flows` package continues to contain public operations in
this release. Renaming it to `operations` or adding a separate `pipelines`
package is outside this page.

## Review Checklist

Before adding a public callable, state:

1. What work it performs.
2. What it accepts and returns.
3. Whether it is one operation or combines several operations.
4. Why its verb describes the returned result.

If those answers are unclear, define the behavior before choosing a public
name.
