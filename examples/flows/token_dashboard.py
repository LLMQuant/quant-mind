"""Example: display used and remaining session tokens after flow runs."""

import asyncio

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import RawText
from quantmind.flows import SessionTokenDashboard, paper_flow

EXAMPLE_TEXT = """
Momentum and value factors remain the most persistent cross-sectional
signals in global equity markets, but turnover-aware portfolio
construction is required to retain net alpha after implementation costs.
"""


async def main() -> None:
    """Run two flow calls and print a session token usage dashboard."""
    dashboard = SessionTokenDashboard(session_token_budget=50_000)
    cfg = PaperFlowCfg(model="gpt-4o-mini")

    await paper_flow(
        RawText(text=EXAMPLE_TEXT),
        cfg=cfg,
        extra_run_hooks=[dashboard],
    )

    await paper_flow(
        RawText(text=EXAMPLE_TEXT),
        cfg=cfg,
        extra_run_hooks=[dashboard],
        extra_instructions="Summarize in fewer than 120 words.",
    )

    print(dashboard.render())


if __name__ == "__main__":
    asyncio.run(main())
