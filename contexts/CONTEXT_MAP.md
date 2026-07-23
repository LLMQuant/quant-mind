# Context Map

## Quick Summary

- **Purpose**: The navigation index for `contexts/`. Before reading or writing anything under `contexts/`, start here to find the one page a task needs.
- **Read when**: Beginning any repository, design, or library-usage task, or adding a new `contexts/` page (register it in the map below and in its area index).
- **Load next**: Pick the single route in [Where to Start](#where-to-start); then open only that page and follow its `Load next` line.
- **Status**: Current. `AGENTS.md` (imported by `CLAUDE.md`) points here as the first read for `contexts/` work.

## Contents

- [Directory Map](#directory-map)
- [Where to Start](#where-to-start)

## Directory Map

```
contexts/
├── CONTEXT_MAP.md              ← you are here: the navigation index
├── README.md                   ← routing entry point (dev / usage / design)
├── design/                     ← accepted design decisions and planned behavior
│   ├── README.md               ← design index
│   ├── positioning.md          ← canonical positioning: workbench, V1/V2, eval
│   ├── flow/
│   │   ├── paper.md            ← source-first Paper Flow V1
│   │   └── news.md             ← news collection design and behavior
│   ├── knowledge/paper.md      ← Paper source and artifact models
│   ├── library/local.md        ← LocalKnowledgeLibrary storage and retrieval
│   ├── mind/retrieval.md       ← page-preserving structure tree + agentic retrieval
│   ├── operations/
│   │   ├── naming.md           ← public operation naming rules
│   │   └── orchestration.md    ← pipelines vs components (altitude)
│   ├── preprocess/pdf.md       ← page-aware ParsedDocument
│   ├── rag/document.md         ← deterministic chunking + BM25
│   └── utils/
│       ├── structured_output.md ← structured-output contract
│       └── usage.md            ← per-run token / time / step usage
├── dev/                        ← contributor and agent development rules
│   ├── README.md               ← dev rule index
│   ├── labels.md               ← issue / PR label taxonomy
│   ├── github-writing.md       ← GitHub prose style (no hard-wrap)
│   └── harness-engineering.md  ← enforcement layers, hooks, rules, alignment
└── usage/
    └── README.md               ← using QuantMind as a library
```

## Where to Start

| I want to | Open |
|---|---|
| Understand the whole `contexts/` layout | the [Directory Map](#directory-map) above |
| Understand what QuantMind is (positioning, V1/V2) | [`design/positioning.md`](design/positioning.md) |
| Develop, fix, test, or review code | [`dev/README.md`](dev/README.md) |
| Use QuantMind as a library | [`usage/README.md`](usage/README.md) |
| Read or change a design decision | [`design/README.md`](design/README.md) |
| Add a hook, rule, or CI gate | [`dev/harness-engineering.md`](dev/harness-engineering.md) |
