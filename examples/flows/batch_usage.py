"""Fan out over papers, then read aggregate token usage and priced cost.

``batch_run`` now reports the SDK token usage it consumed and, when given a
price table, an estimated USD cost — and enforces the ``cfg`` budget
guardrails, marking any input skipped after the budget trips with a
``BudgetExceededError``. Requires ``OPENAI_API_KEY`` (like the other flow
examples).
"""

import asyncio

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import ArxivIdentifier
from quantmind.flows import (
    BudgetExceededError,
    PriceRate,
    batch_run,
    paper_flow,
)


async def main() -> None:
    """Build several papers under one shared budget and report spend."""
    cfg = PaperFlowCfg(
        model="gpt-4o-mini",
        max_total_input_tokens=200_000,  # stop launching once we cross this
    )
    # Caller-supplied pricing (USD per 1M tokens); the library ships none.
    prices = {
        "gpt-4o-mini": PriceRate(input_usd_per_1m=0.15, output_usd_per_1m=0.60),
    }
    inputs = [
        ArxivIdentifier(id="1706.03762v7"),
        ArxivIdentifier(id="2404.11584"),
    ]

    result = await batch_run(
        paper_flow, inputs, cfg=cfg, concurrency=2, prices=prices
    )

    print(f"success={result.success_count} failure={result.failure_count}")
    print(f"tokens={result.tokens_total}")
    print(f"cost_usd≈{result.cost_estimate_usd:.4f}")
    skipped = [
        i for i, e in result.errors if isinstance(e, BudgetExceededError)
    ]
    if skipped:
        print(f"budget-skipped inputs: {skipped}")


if __name__ == "__main__":
    asyncio.run(main())
