"""03 — Read trajectory archive offline.

After 01 / 02 have run, this script needs no network and no npx — it
opens ``runs.jsonl`` and aggregates the ``RunRecord`` fields so you
can see exactly what ``MemoryRunHooks`` captured: timing, token usage,
LLM call breakdown, tool calls.

This is also the right shape for building dashboards / cost reports
later: each line is one self-contained run record.

Run:
    python examples/memory/03_inspect_trajectory.py ./.qm-memory
"""

import json
import sys
from pathlib import Path


def main() -> None:
    memory_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "./.qm-memory")
    runs_jsonl = memory_dir / "runs.jsonl"
    if not runs_jsonl.exists():
        print(f"No {runs_jsonl}. Run 01_basic.py or 02_serial_loop.py first.")
        sys.exit(1)

    records = [
        json.loads(line) for line in runs_jsonl.read_text().splitlines() if line
    ]
    print(f"Loaded {len(records)} run record(s) from {runs_jsonl}\n")

    # Per-run summary.
    for r in records:
        toks = r["tokens_total"]
        print(
            f"  {r['run_id']}  "
            f"workflow={r['workflow_name']}  "
            f"duration={r['duration_seconds']:.2f}s  "
            f"in={toks['input']}/out={toks['output']}  "
            f"llm_calls={len(r['llm_calls'])}  "
            f"tool_calls={len(r['tool_calls'])}  "
            f"error={r['error']!r}"
        )

    # Aggregate.
    total_in = sum(r["tokens_total"]["input"] for r in records)
    total_out = sum(r["tokens_total"]["output"] for r in records)
    total_dur = sum(r["duration_seconds"] for r in records)
    n_failed = sum(1 for r in records if r["error"] is not None)
    print()
    print(
        f"Aggregate: tokens_in={total_in} tokens_out={total_out} "
        f"total_duration={total_dur:.2f}s "
        f"failed={n_failed}/{len(records)}"
    )


if __name__ == "__main__":
    main()
