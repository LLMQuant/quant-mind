"""02 — Serial loop sharing one FilesystemMemory.

The point of cross-step memory is letting run N see what runs 1..N-1
left behind. ``FilesystemMemory`` exposes ``notes/`` and ``items/`` to
the Agent through an MCP filesystem server, so the Agent can list /
read / write files there during its own turn.

What to look for:

- ``./.qm-memory/runs/`` accumulates one trajectory record per loop
  iteration (3 here).
- ``./.qm-memory/workspace/notes/`` may grow if the Agent decides to
  write notes while extracting (it sees the seeded ``README.md`` and is
  encouraged to do so).
- ``./.qm-memory/runs.jsonl`` has 3 appended lines after this script.

This is the "memory-accumulating workflow" pattern. ``batch_run``
rejects ``memory=`` at the signature level — for memory accumulation
you write the loop yourself, exactly like below.

Prerequisites: OPENAI_API_KEY, Node.js + npx, network.

Run:
    python examples/memory/02_serial_loop.py
"""

import asyncio
from pathlib import Path

from quantmind.configs.paper import RawText
from quantmind.flows import paper_flow
from quantmind.mind.memory import FilesystemMemory

_FAKE_PAPERS = [
    (
        "# Momentum returns 1\n"
        "Title: Cross-sectional momentum on US equities\n"
        "Body: Long winners short losers monthly."
    ),
    (
        "# Momentum returns 2\n"
        "Title: Time-series momentum on commodities\n"
        "Body: 12-month look-back, monthly rebalance."
    ),
    (
        "# Momentum returns 3\n"
        "Title: Combining XS and TS momentum\n"
        "Body: 50/50 equal weight composite."
    ),
]


async def main() -> None:
    mem = FilesystemMemory(Path("./.qm-memory"))

    for idx, body in enumerate(_FAKE_PAPERS, start=1):
        print(f"\n=== run {idx}/3 ===")
        paper = await paper_flow(RawText(text=body), memory=mem)
        print(f"  -> {paper.nodes[paper.root_node_id].title!r}")

    # Quick post-run summary.
    runs_dir = mem.memory_dir / "runs"
    print(
        f"\nTrajectory files: {len(list(runs_dir.glob('*.json')))} in {runs_dir}"
    )

    notes_dir = mem.workspace / "notes"
    notes = list(notes_dir.glob("*"))
    print(f"Notes the agent left: {len(notes)} in {notes_dir}")
    for n in notes[:5]:
        print(f"  - {n.name}")


if __name__ == "__main__":
    asyncio.run(main())
