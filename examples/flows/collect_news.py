"""Collect one replayable day of PR Newswire observations."""

import asyncio
from datetime import datetime, timedelta, timezone

from quantmind.configs import NewsCollectionCfg, NewsWindow
from quantmind.flows import collect_news


async def main() -> None:
    """Collect and summarize the preceding 24-hour window."""
    end = datetime.now(timezone.utc)
    batch = await collect_news(
        NewsWindow(
            source="pr-newswire",
            start=end - timedelta(days=1),
            end=end,
        ),
        cfg=NewsCollectionCfg(retain_raw_html=False),
    )
    print(
        f"observed={batch.observed_count} documents={batch.success_count} "
        f"failures={batch.failure_count} complete={batch.complete}"
    )


if __name__ == "__main__":
    asyncio.run(main())
