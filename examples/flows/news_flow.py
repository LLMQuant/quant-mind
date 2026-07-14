"""Collect one replayable day of PR Newswire observations."""

import asyncio
from datetime import datetime, timedelta, timezone

from quantmind.configs import NewsFlowCfg, NewsWindow
from quantmind.flows import news_flow


async def main() -> None:
    """Collect and summarize the preceding 24-hour window."""
    end = datetime.now(timezone.utc)
    batch = await news_flow(
        NewsWindow(
            source="pr-newswire",
            start=end - timedelta(days=1),
            end=end,
        ),
        cfg=NewsFlowCfg(retain_raw_html=False),
    )
    print(
        f"documents={batch.success_count} failures={batch.failure_count} "
        f"complete={batch.complete}"
    )


if __name__ == "__main__":
    asyncio.run(main())
