"""01 — The shortest memory run.

Builds a ``FilesystemMemory``, runs ``paper_flow`` once with it, then
prints what the trajectory archive looks like on disk afterwards.

What to look for:

- ``./.qm-memory/workspace/`` is created with ``notes/`` ``items/`` and
  a seeded ``README.md`` (the agent's own usage guide for the dir).
- One ``./.qm-memory/runs/<run_id>.json`` file lands per
  ``paper_flow`` call.
- ``./.qm-memory/runs.jsonl`` gets one new line per call (a
  denormalised index of the per-run files — handy for `jq` / `pandas`).

Prerequisites: OPENAI_API_KEY in env, Node.js + npx on PATH.

Run:
    python examples/memory/01_basic.py
"""

import asyncio
from pathlib import Path

from quantmind.configs.paper import RawText
from quantmind.flows import paper_flow
from quantmind.mind.memory import FilesystemMemory


async def main() -> None:
    memory_dir = Path("./.qm-memory")
    mem = FilesystemMemory(memory_dir)

    # A tiny paper-shaped input keeps cost low; swap in
    # ``ArxivIdentifier(id="...")`` to see a real extraction end to end.
    paper = await paper_flow(
        RawText(
            text=(
                "# Toy paper\n\n"
                "Title: Cross-sectional momentum, simplified\n"
                "Author: Demo\n\n"
                "Body: We illustrate momentum on a 3-stock universe."
            )
        ),
        memory=mem,
    )
    print(f"Extracted paper: {paper.title!r}")

    # Show what landed under .qm-memory/.
    print(f"\nMemory layout under {memory_dir}:")
    for child in sorted(memory_dir.rglob("*")):
        if child.is_file():
            print(
                f"  {child.relative_to(memory_dir)}  ({child.stat().st_size}B)"
            )

    runs_jsonl = memory_dir / "runs.jsonl"
    if runs_jsonl.exists():
        print(f"\nLast line of {runs_jsonl}:")
        print(runs_jsonl.read_text().splitlines()[-1][:200] + " ...")


if __name__ == "__main__":
    asyncio.run(main())
