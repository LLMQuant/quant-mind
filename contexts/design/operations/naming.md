# Public Operation Names

## Quick Summary

- **Purpose**: Define clear names for public QuantMind callables and their input, config, and result types.
- **Read when**: Adding or renaming a public function, service, operation type, or pipeline.
- **Status**: Use these rules for new names. The legacy paper extraction function has been removed; the paper semantic build is now `PaperFlow(PaperSemanticCfg(...)).build(input)`, so no `*_flow` functions remain.
- **Core rule**: Name a function or service method with a verb that says what it does. Use `pipeline` only when one callable combines several public operations. Do not use `flow` as a **verb**; `flow` as a **noun** naming a finished pipeline collection (`PaperFlow`) is allowed.

## Contents

- [Scope](#scope)
- [Name Patterns](#name-patterns)
- [Service Class Patterns](#service-class-patterns)
- [Config-Bound Flow Patterns](#config-bound-flow-patterns)
- [Type Names](#type-names)
- [Current API](#current-api)
- [Review Checklist](#review-checklist)

## Scope

This page defines how to name public QuantMind functions and small service classes. It does not rename packages or existing callables. Any existing name change needs a separate plan for users who depend on the old name.

The runtime library API is designed for Python callers. Repository guidance for coding agents belongs in `AGENTS.md`, skills, docs, fixtures, and checks; it does not belong in the runtime API.

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

`flow` is not a verb that explains a result. Do not add a public `*_flow` **function** name only because a function performs several steps. Name the result precisely, or use `*_pipeline` when the function combines multiple public operations.

`Flow` as a **noun**, on a domain object that binds a config and applies the finished pipeline to each input, is allowed and preferred there: `PaperFlow(cfg)` binds the settings once; `build(input)` applies them to each input. The object reads as "the paper's workflow," its *method* uses an intent verb (`build`), and the cfg *type* selects which knowledge shape it produces. What stays banned is `flow` as a verb or a `do_x_flow()` function name.

## Service Class Patterns

Use a service class only when repeated calls share dependencies, policy, or a lifecycle that belongs at construction time. Keep the active input and result on method arguments and return values rather than mutable instance state.

- Name the class by its stable role (a `*Builder` / `*Store` noun), not by a single method it exposes.
- Name its primary async methods with the same intent verbs used for functions, such as `build()` or `retrieve()`.
- Do not add a class only to group helpers or shorten one function signature.
- Do not put model clients, stores, or runtime policy on frozen knowledge artifacts; services consume and return those canonical values.
- Do not add a base service, registry, or manager hierarchy around one concrete implementation.
- **Config binding alone justifies a class.** A class earns its keep by binding the immutable `cfg` that must stay constant across a batch, even with no store or provider to hold. `AgenticRetriever(RetrievalCfg(...))` binds the retrieval config once; `retrieve(structure, question)` takes only the operand. Do not demote such a class to a function just because it holds no external dependency.

## Config-Bound Flow Patterns

A domain flow binds an immutable `cfg` at construction and applies it to each input, so a batch runs under one unified, reproducible setting:

- Name it `<Domain>Flow` (`PaperFlow`). `Flow` here is a noun for "the domain's workflow"; the cfg *type* selects which knowledge shape it builds.
- Construct with the cfg (`PaperFlow(PaperStructureCfg(...))`); the build method takes only the input (`build(input)`), never a per-call cfg.
- Use an intent verb for the method (`build`). The flow binds no store and does no persistence or retrieval — those are downstream of the artifact it returns.
- `batch_run(flow.build, inputs)` is the intended batch shape: one bound cfg, many inputs, one setting.

## Type Names

- Input types describe what the caller supplies, such as `NewsWindow` or `PaperInput`.
- Config types name the domain and stage, such as `NewsCollectionCfg`, `PaperStructureCfg`, or `RetrievalCfg`.
- Result types describe returned data, such as `NewsBatch`, `PaperSemanticResult`, or `PageIndex`.
- Keep provider names out of public function names unless callers are choosing provider-specific behavior.

## Current API

- `collect_news` is a collection operation and follows these naming rules.
- `batch_run` is a generic batch helper, not a news or paper operation.
- The `*_flow` function-name pattern is banned for new operations; the former paper extraction function was removed rather than kept as a wrapper.
- `PaperFlow(cfg)` is the config-bound paper flow and the single entry point for every paper shape; `build(input)` produces a knowledge artifact whose shape is chosen by the cfg *type* (`PaperStructureCfg` → `PaperStructureTree`, `PaperSemanticCfg` → `PaperSemanticResult`). `Flow` as a noun here is deliberate and allowed.
- `AgenticRetriever(cfg)` is the reasoning-retrieval service in `quantmind.mind`; `retrieve(structure, question)` returns evidence values. It has one behavior — an LLM agent reasons over the structure — so it binds `RetrievalCfg` but does no cfg-type dispatch. Mechanical semantic search is `quantmind.library.search`, a different layer, not a strategy here.

The current `quantmind.flows` package continues to contain public operations in this release. Renaming it to `operations` or adding a separate `pipelines` package is outside this page.

## Review Checklist

Before adding a public callable, state:

1. What work it performs.
2. What it accepts and returns.
3. Whether it is one operation or combines several operations.
4. Why its verb describes the returned result.
5. If it is a service class, which dependencies, policy, or lifecycle are reused across calls and why they should not remain explicit function arguments.

If those answers are unclear, define the behavior before choosing a public name.
