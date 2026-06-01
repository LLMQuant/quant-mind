# `mind/memory` examples

Small runnable scripts that walk you through `FilesystemMemory` +
`MemoryRunHooks` end to end. Each one is self-contained — read the
top-of-file comment block for what it shows.

## Prerequisites

- `pip install -e ".[dev]"` (so `quantmind` is importable).
- `OPENAI_API_KEY` set in your shell (the real Agent run uses an LLM).
- Node.js + `npx` on PATH — `FilesystemMemory` launches
  `@modelcontextprotocol/server-filesystem` over stdio. Examples 01,
  02, and 04 spawn a real MCP subprocess; example 03 only reads disk.

## What each script demonstrates

| # | Script | What you learn |
|---|--------|----------------|
| 01 | `01_basic.py` | The shortest possible memory run: build a `FilesystemMemory`, pass it to `paper_flow`, then look at the disk layout that gets created (`workspace/notes/`, `workspace/items/`, `runs/`, `runs.jsonl`). |
| 02 | `02_serial_loop.py` | Process several inputs in a serial `for` loop sharing one `FilesystemMemory`. Each run can read what previous runs left in `workspace/notes/` via the MCP filesystem server. (`batch_run` rejects `memory=` by design — for memory-accumulating workflows, you write the loop yourself.) |
| 03 | `03_inspect_trajectory.py` | Disk-only analysis: open `runs.jsonl`, parse the `RunRecord` JSON, and aggregate tokens / durations across runs. No network or `npx` needed. |
| 04 | `04_custom_run_hooks.py` | Compose your own `RunHooks` (e.g., a custom logger) alongside the built-in `MemoryRunHooks` via the `extra_run_hooks=` kwarg. Both hooks fire for every lifecycle event in the order you registered them. |

## Run them

```bash
# from the repo root, after installing
python examples/memory/01_basic.py
python examples/memory/02_serial_loop.py
python examples/memory/03_inspect_trajectory.py ./.qm-memory
python examples/memory/04_custom_run_hooks.py
```

All four examples write under `./.qm-memory/` (a directory created
under your current working directory). Delete it freely between runs;
nothing is shared with your other projects.

`FilesystemMemory` exposes only `./.qm-memory/workspace/` to the Agent.
The trajectory archive remains system-managed under `./.qm-memory/runs/`
and `./.qm-memory/runs.jsonl`.

## A note on `cfg.archive_trajectory`

`FilesystemMemory.run_hooks()` returns the `MemoryRunHooks` instance
that produces `runs/<run_id>.json` and the `runs.jsonl` index. The
runner only attaches it when `cfg.archive_trajectory` is `True`
(the default). If you set it to `False`, the agent still gets the
`mcp_servers` + `tools` from your `Memory` (so it can read previous
notes) — only the trajectory archive is suppressed.
