#!/usr/bin/env python3
"""Run the bounded live-network checks for public news collection."""

import asyncio
import re
from datetime import datetime, timedelta, timezone

from quantmind.preprocess import NewsDocument
from quantmind.preprocess.fetch.http import FetchPolicy, HttpFetcher
from quantmind.preprocess.fetch.rss import fetch_rss_feed
from quantmind.preprocess.pr_newswire import (
    PRNewswireDiscovery,
    _collect_observation,
    _discover_pr_newswire,
)

_PUBLIC_RSS_URL = "https://www.prnewswire.com/rss/news-releases-list.rss"
_RSS_TIMEOUT_SECONDS = 60
_DISCOVERY_TIMEOUT_SECONDS = 300
_ARTICLE_SAMPLE_TIMEOUT_SECONDS = 180
_ARTICLE_SAMPLE_SIZE = 25
_LIVE_FETCH_POLICY = FetchPolicy(
    max_attempts=3,
    backoff_base_seconds=0.5,
    backoff_max_seconds=5.0,
    jitter_seconds=0.2,
    max_concurrency_per_host=2,
    min_interval_seconds=0.25,
)

# This oracle intentionally stays independent from the production extractor.
# It checks the first supported symbol after each explicit exchange prefix. It
# does not infer later list members or exchanges outside the current whitelist.
_CONTROL_EXCHANGE_TICKER_RE = re.compile(
    r"(?:\(|\b)"
    r"(?:NASDAQ|NYSE(?:\s+American|\s+MKT|\s+Arca)?|"
    r"AMEX|OTCQX|OTCQB|OTC|CBOE)"
    r"\s*:\s*"
    r"(?:"
    r"\[(?P<link>[A-Z][A-Z0-9.-]{0,9})\]\([^\)\n]*\)"
    r"|\*{1,2}(?P<emphasis>[A-Z][A-Z0-9.-]{0,9})\*{1,2}"
    r"|(?P<plain>[A-Z][A-Z0-9.-]{0,9})"
    r")",
    re.IGNORECASE,
)


async def _check_rss() -> bool:
    try:
        async with HttpFetcher(
            policy=_LIVE_FETCH_POLICY,
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


async def _check_discovery(
    start: datetime,
    end: datetime,
) -> tuple[bool, PRNewswireDiscovery | None]:
    try:
        result = await asyncio.wait_for(
            _discover_pr_newswire(start=start, end=end),
            timeout=_DISCOVERY_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        print(f"[FAIL] pr-newswire-discovery: {type(exc).__name__}: {exc}")
        return False, None

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
    return passed, result if passed else None


def _ticker_hint_control_counts(
    documents: tuple[NewsDocument, ...],
) -> tuple[int, int]:
    expected_count = 0
    recovered_count = 0
    for document in documents:
        expected_symbols = {
            (
                match.group("link")
                or match.group("emphasis")
                or match.group("plain")
            ).upper()
            for match in _CONTROL_EXCHANGE_TICKER_RE.finditer(
                document.cleaned_markdown
            )
        }
        actual_symbols = {hint.symbol for hint in document.ticker_hints}
        expected_count += len(expected_symbols)
        recovered_count += len(expected_symbols & actual_symbols)
    return expected_count, recovered_count


async def _check_ticker_hints(discovery: PRNewswireDiscovery) -> bool:
    sample = []
    seen_urls: set[str] = set()
    for observation in discovery.observations:
        if observation.canonical_url in seen_urls:
            continue
        sample.append(observation)
        seen_urls.add(observation.canonical_url)
        if len(sample) == _ARTICLE_SAMPLE_SIZE:
            break

    if not sample:
        print("[FAIL] pr-newswire-ticker-hints: no article URLs to sample")
        return False

    try:
        async with HttpFetcher(
            policy=_LIVE_FETCH_POLICY,
            timeout=30.0,
            max_bytes=10_000_000,
        ) as fetcher:
            outcomes = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        _collect_observation(
                            observation,
                            fetcher=fetcher,
                            retain_raw_html=False,
                        )
                        for observation in sample
                    )
                ),
                timeout=_ARTICLE_SAMPLE_TIMEOUT_SECONDS,
            )
    except Exception as exc:
        print(f"[FAIL] pr-newswire-ticker-hints: {type(exc).__name__}: {exc}")
        return False

    documents = tuple(
        outcome for outcome in outcomes if isinstance(outcome, NewsDocument)
    )
    expected_count, recovered_count = _ticker_hint_control_counts(documents)
    article_failure_count = len(outcomes) - len(documents)
    recall = recovered_count / expected_count if expected_count else 0.0
    passed = bool(documents) and expected_count > 0 and recall == 1.0
    state = "PASS" if passed else "FAIL"
    print(
        f"[{state}] pr-newswire-ticker-hints: sampled={len(sample)} "
        f"parsed={len(documents)} failures={article_failure_count} "
        f"expected={expected_count} recovered={recovered_count} "
        f"recall={recall:.1%}"
    )
    return passed


async def main(*, now: datetime | None = None) -> int:
    """Run all live checks and return a process exit code."""
    end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    end = end.replace(second=0, microsecond=0)
    start = end - timedelta(days=1)
    print(f"news window: [{start.isoformat()}, {end.isoformat()})")

    rss_passed = await _check_rss()
    discovery_passed, discovery = await _check_discovery(start, end)
    if discovery is None:
        print("[FAIL] pr-newswire-ticker-hints: discovery unavailable")
        ticker_hints_passed = False
    else:
        ticker_hints_passed = await _check_ticker_hints(discovery)
    return 0 if all((rss_passed, discovery_passed, ticker_hints_passed)) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
