#!/usr/bin/env python3
"""Run the bounded live-network checks for public news collection."""

import asyncio
from datetime import datetime, timedelta, timezone

from quantmind.preprocess.fetch.http import FetchPolicy, HttpFetcher
from quantmind.preprocess.fetch.rss import fetch_rss_feed
from quantmind.preprocess.pr_newswire import _discover_pr_newswire

_PUBLIC_RSS_URL = "https://www.prnewswire.com/rss/news-releases-list.rss"
_RSS_TIMEOUT_SECONDS = 60
_DISCOVERY_TIMEOUT_SECONDS = 300


async def _check_rss() -> bool:
    try:
        async with HttpFetcher(
            policy=FetchPolicy(
                max_attempts=3,
                backoff_base_seconds=0.5,
                backoff_max_seconds=5.0,
                jitter_seconds=0.2,
                max_concurrency_per_host=1,
                min_interval_seconds=0.25,
            ),
            timeout=30.0,
            max_bytes=5_000_000,
        ) as fetcher:
            feed = await asyncio.wait_for(
                fetch_rss_feed(_PUBLIC_RSS_URL, fetcher=fetcher),
                timeout=_RSS_TIMEOUT_SECONDS,
            )
    except Exception as exc:
        print(f"[FAIL] rss: {type(exc).__name__}: {exc}")
        return False

    usable_count = sum(
        bool(item.title.strip() and (item.url or "").strip())
        for item in feed.items
    )
    passed = bool(feed.items) and usable_count > 0
    state = "PASS" if passed else "FAIL"
    print(
        f"[{state}] rss: items={len(feed.items)} usable={usable_count} "
        f"url={_PUBLIC_RSS_URL}"
    )
    return passed


async def _check_discovery(start: datetime, end: datetime) -> bool:
    try:
        result = await asyncio.wait_for(
            _discover_pr_newswire(start=start, end=end),
            timeout=_DISCOVERY_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        print(f"[FAIL] pr-newswire-discovery: {type(exc).__name__}: {exc}")
        return False

    urls = [observation.canonical_url for observation in result.observations]
    duplicate_count = len(urls) - len(set(urls))
    in_window_count = sum(
        start <= observation.published_at < end
        for observation in result.observations
    )
    passed = (
        bool(result.observations)
        and in_window_count == len(result.observations)
        and result.complete
        and not result.failures
    )
    state = "PASS" if passed else "FAIL"
    print(
        f"[{state}] pr-newswire-discovery: "
        f"observed={len(result.observations)} "
        f"in_window={in_window_count} "
        f"duplicates={duplicate_count} pages={result.page_count} "
        f"failures={len(result.failures)} complete={result.complete}"
    )
    for failure in result.failures[:3]:
        print(
            f"       {failure.stage} {failure.error_type} {failure.source_url}"
        )
    return passed


async def main(*, now: datetime | None = None) -> int:
    """Run all live checks and return a process exit code."""
    end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    end = end.replace(second=0, microsecond=0)
    start = end - timedelta(days=1)
    print(f"news window: [{start.isoformat()}, {end.isoformat()})")

    rss_passed = await _check_rss()
    discovery_passed = await _check_discovery(start, end)
    return 0 if rss_passed and discovery_passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
