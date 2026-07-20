# Public Operation Names

## Quick Summary

- **Purpose**: Define clear names for public QuantMind callables and their input, config, and result types.
- **Read when**: Adding or renaming a public function, service, operation type, or pipeline.
- **Status**: Use these rules for new names. Keep the existing `paper_flow` name until a separate change explains how callers will migrate.
- **Core rule**: Name a function or service method with a verb that says what it does. Use `pipeline` only when one callable combines several public operations. Do not use `flow` as a verb.

## Contents

- [Scope](#scope)
- [Name Patterns](#name-patterns)
- [Service Class Patterns](#service-class-patterns)
- [Type Names](#type-names)
- [Current API](#current-api)
- [Review Checklist](#review-checklist)

## Scope

This page defines how to name public QuantMind functions and small service
classes. It does not rename packages or existing callables. Any existing name
change needs a separate plan for users who depend on the old name.

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

## Service Class Patterns

Use a service class only when repeated calls share dependencies, policy, or a
lifecycle that belongs at construction time. Keep the active input and result
on method arguments and return values rather than mutable instance state.

- Name the class by its stable role, such as `PaperStructureBuilder` or
  `StructureRetriever`.
- Name its primary async methods with the same intent verbs used for functions,
  such as `build()` or `retrieve()`.
- Do not add a class only to group helpers or shorten one function signature.
- Do not put model clients, stores, or runtime policy on frozen knowledge
  artifacts; services consume and return those canonical values.
- Do not add a base service, registry, or manager hierarchy around one concrete
  implementation.

## Type Names

- Input types describe what the caller supplies, such as `NewsWindow` or
  `PaperInput`.
- Config types name the domain and stage, such as `NewsCollectionCfg`,
  `PaperStructureCfg`, or `RetrievalCfg`.
- Result types describe returned data, such as `NewsBatch`, `PaperFlowResult`, or
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
5. If it is a service class, which dependencies, policy, or lifecycle are
   reused across calls and why they should not remain explicit function
   arguments.

If those answers are unclear, define the behavior before choosing a public
name.
