#!/usr/bin/env python3
"""Run the manual live-network checklist for PR Newswire."""

import asyncio

from quantmind.preprocess import (
    PR_NEWSWIRE,
    FetchPolicy,
    WireFeedConfig,
    fetch_wire_documents,
)

_PUBLIC_FEED = "https://www.prnewswire.com/rss/news-releases-list.rss"


async def _check_pr_newswire() -> bool:
    provider_name = PR_NEWSWIRE.name
    try:
        result = await asyncio.wait_for(
            fetch_wire_documents(
                WireFeedConfig(
                    provider=PR_NEWSWIRE,
                    feed_urls=(_PUBLIC_FEED,),
                    fetch_policy=FetchPolicy(
                        max_attempts=2,
                        min_interval_seconds=0.25,
                    ),
                )
            ),
            timeout=120,
        )
    except Exception as exc:
        print(f"[FAIL] {provider_name}: {type(exc).__name__}: {exc}")
        return False

    documents = result.documents
    checks = {
        "documents": bool(documents),
        "no_failures": not result.failures,
        "raw_feed_entry": bool(documents)
        and all(document.raw_feed_entry.bytes for document in documents),
        "raw_article": bool(documents)
        and all(
            document.raw_article is not None
            and bool(document.raw_article.bytes)
            for document in documents
        ),
        "cleaned_markdown": bool(documents)
        and all(document.cleaned_markdown.strip() for document in documents),
        "unique_identity": len({document.identity for document in documents})
        == len(documents),
    }
    for check, passed in checks.items():
        state = "PASS" if passed else "FAIL"
        print(f"[{state}] {provider_name}: {check}")
    print(
        f"       documents={result.success_count} "
        f"failures={result.failure_count}"
    )
    for failure in result.failures[:3]:
        print(
            f"       {failure.stage} {failure.error_type} {failure.source_url}"
        )
    return all(checks.values())


async def main() -> int:
    """Run the PR Newswire checklist and return a process exit code."""
    return 0 if await _check_pr_newswire() else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
