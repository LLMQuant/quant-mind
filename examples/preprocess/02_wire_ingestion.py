"""Fetch the public PR Newswire feed and print a compact summary."""

import asyncio

from quantmind.preprocess import (
    PR_NEWSWIRE,
    FetchPolicy,
    WireFeedConfig,
    fetch_wire_documents,
)

_PR_NEWSWIRE_FEED = "https://www.prnewswire.com/rss/news-releases-list.rss"


async def main() -> None:
    """Fetch PR Newswire and print a compact result summary."""
    result = await fetch_wire_documents(
        WireFeedConfig(
            provider=PR_NEWSWIRE,
            feed_urls=(_PR_NEWSWIRE_FEED,),
            fetch_policy=FetchPolicy(min_interval_seconds=0.25),
        )
    )
    print(f"documents={result.success_count} failures={result.failure_count}")
    if result.documents:
        first = result.documents[0]
        print(first.title)
        print(first.identity)


if __name__ == "__main__":
    asyncio.run(main())
