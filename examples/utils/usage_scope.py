"""Measure token usage, time, and LLM steps for one flow run.

``usage_scope`` wraps any flow call and, after the block, reports how many
tokens it spent, how long it took, and how many model calls it made — all read
from the traces the Agents SDK already emits. Tokens, time, and steps only; no
cost. Read ``run.usage`` after the ``with`` block, not inside it.

Here it wraps ``PaperFlow(PaperStructureCfg).build`` (one structure-extraction
agent). ``requests`` is the model-call count: OpenRouter accepts the strict
``json_schema`` request, so this is a single clean call and ``requests`` is 1.
It grows when the provider or flow does more — a json-object-only provider adds
a rejected strict attempt before its json_object fallback (``requests`` 2), and
a fan-out flow such as the summary map-reduce shows one step per researcher plus
the reducer, where ``wall`` drops below ``busy`` as they overlap.

Running this end to end needs network access and the chosen provider's API key.
The example imports and type-checks offline.

Example output (a real run on OpenRouter deepseek-v4-flash; varies by run):

    built structure tree: 10 nodes
    tokens: in=1144 out=429 total=1573
    llm-steps (requests): 1
    time: wall=13.07s busy=13.07s
      - paper_structure_builder  openrouter/deepseek/deepseek-v4-flash in=1144 out=429 13.07s
"""

import asyncio
import sys
from pathlib import Path

from quantmind.configs import PaperStructureCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import PaperFlow
from quantmind.utils.usage import usage_scope

# usage_scope works with any provider; swap in your model and set its API key.
# OpenRouter accepts strict json_schema, so this run is one clean call. A
# json-object-only provider (e.g. "litellm/deepseek/deepseek-chat") instead adds
# a rejected strict attempt before the json_object fallback (requests=2).
_MODEL = "litellm/openrouter/deepseek/deepseek-v4-flash"


async def main(pdf_path: Path) -> None:
    """Build one paper structure tree and print the per-run usage of that flow."""
    flow = PaperFlow(PaperStructureCfg(model=_MODEL))

    with usage_scope("quantmind.paper.structure") as run:
        tree = await flow.build(LocalFilePath(path=pdf_path))

    print(f"built structure tree: {len(tree.nodes)} nodes")
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
            f"  - {step.label:<24} {step.model or '-':<24} "
            f"in={step.input_tokens} out={step.output_tokens} "
            f"{step.duration_seconds:.2f}s"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python examples/utils/usage_scope.py paper.pdf"
        )
    asyncio.run(main(Path(sys.argv[1])))
