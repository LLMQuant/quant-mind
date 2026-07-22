"""Measure token usage, time, and LLM steps for one flow run.

``usage_scope`` wraps any flow call and, after the block, reports how many
tokens it spent, how long it took, and how many model calls it made — all read
from the traces the Agents SDK already emits. Tokens, time, and steps only; no
cost. Read ``run.usage`` after the ``with`` block, not inside it.

Running this end to end needs network access (a model provider). The example
imports and type-checks offline.
"""

import asyncio
import sys
from pathlib import Path

from quantmind.configs import PaperStructureCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import PaperFlow
from quantmind.utils.usage import usage_scope


async def main(pdf_path: Path) -> None:
    """Build one paper structure tree and print its per-run usage."""
    flow = PaperFlow(PaperStructureCfg(model="gpt-5.6-luna"))

    with usage_scope("quantmind.paper.structure") as run:
        tree = await flow.build(LocalFilePath(path=pdf_path))

    print("built:", tree.id)
    usage = run.usage
    print(
        f"tokens: in={usage.input_tokens} out={usage.output_tokens} "
        f"total={usage.total_tokens}"
    )
    print(f"llm-steps (requests): {usage.requests}")
    print(
        f"time: wall={usage.wall_seconds:.2f}s busy={usage.busy_seconds:.2f}s"
    )
    for step in usage.steps:
        print(
            f"  - {step.label} [{step.model}] "
            f"in={step.input_tokens} out={step.output_tokens} "
            f"{step.duration_seconds:.2f}s"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python examples/utils/usage_scope.py paper.pdf"
        )
    asyncio.run(main(Path(sys.argv[1])))
