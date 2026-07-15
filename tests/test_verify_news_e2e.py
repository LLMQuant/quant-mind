import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from quantmind.preprocess import (
    NewsArtifact,
    NewsDocument,
    NewsTickerHint,
)
from scripts import verify_news_e2e

_PUBLISHED_AT = datetime(2026, 7, 14, tzinfo=timezone.utc)


def _feed_item(*, title: str = "Release", url: str | None = "https://x"):
    return SimpleNamespace(title=title, url=url)


def _discovery_result(
    *,
    urls: tuple[str, ...] = ("https://example.test/release",),
    failures: tuple[object, ...] = (),
    complete: bool = True,
    published_at: datetime = _PUBLISHED_AT,
):
    return SimpleNamespace(
        observations=tuple(
            SimpleNamespace(
                canonical_url=url,
                published_at=published_at,
            )
            for url in urls
        ),
        failures=failures,
        complete=complete,
        page_count=2,
    )


def _document(
    markdown: str,
    *symbols: str,
) -> NewsDocument:
    artifact = NewsArtifact(
        bytes=None,
        content_hash="hash",
        content_type="text/html",
        source_url="https://example.test/release",
        resolved_url="https://example.test/release",
        status_code=200,
    )
    return NewsDocument(
        source="pr-newswire",
        identity="news:pr-newswire:test",
        cleaned_markdown=markdown,
        content_hash="hash",
        discovery_artifact=artifact,
        article_artifact=artifact,
        ticker_hints=tuple(NewsTickerHint(symbol=symbol) for symbol in symbols),
    )


class VerifyNewsE2ETests(unittest.IsolatedAsyncioTestCase):
    async def test_main_reports_duplicates_and_passes_all_components(self):
        now = datetime(2026, 7, 14, 12, 30, tzinfo=timezone.utc)
        feed = SimpleNamespace(items=(_feed_item(), _feed_item(url=None)))
        discovery = _discovery_result(
            urls=(
                "https://example.test/releases/one",
                "https://example.test/releases/one",
                "https://example.test/releases/two",
            )
        )

        with (
            patch.object(
                verify_news_e2e,
                "fetch_rss_feed",
                new=AsyncMock(return_value=feed),
            ),
            patch.object(
                verify_news_e2e,
                "_discover_pr_newswire",
                new=AsyncMock(return_value=discovery),
            ) as discover,
            patch.object(
                verify_news_e2e,
                "_check_ticker_hints",
                new=AsyncMock(return_value=True),
            ) as ticker_check,
            redirect_stdout(io.StringIO()) as output,
        ):
            exit_code = await verify_news_e2e.main(now=now)

        self.assertEqual(exit_code, 0)
        self.assertIn("[PASS] rss", output.getvalue())
        self.assertIn("duplicates=1", output.getvalue())
        discover.assert_awaited_once_with(
            start=now - timedelta(days=1),
            end=now,
        )
        ticker_check.assert_awaited_once_with(discovery)

    async def test_rss_failure_does_not_skip_discovery(self):
        discovery = _discovery_result()
        with (
            patch.object(
                verify_news_e2e,
                "fetch_rss_feed",
                new=AsyncMock(side_effect=ValueError("bad XML")),
            ),
            patch.object(
                verify_news_e2e,
                "_discover_pr_newswire",
                new=AsyncMock(return_value=discovery),
            ) as discover,
            patch.object(
                verify_news_e2e,
                "_check_ticker_hints",
                new=AsyncMock(return_value=True),
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            exit_code = await verify_news_e2e.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("[FAIL] rss: ValueError: bad XML", output.getvalue())
        discover.assert_awaited_once()

    async def test_rss_rejects_empty_or_wholly_unusable_items(self):
        feeds = (
            SimpleNamespace(items=()),
            SimpleNamespace(items=(_feed_item(url=None),)),
        )
        for feed in feeds:
            with self.subTest(feed=feed):
                with (
                    patch.object(
                        verify_news_e2e,
                        "fetch_rss_feed",
                        new=AsyncMock(return_value=feed),
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    passed = await verify_news_e2e._check_rss()
                self.assertFalse(passed)

    async def test_discovery_rejects_invalid_results(self):
        start = datetime(2026, 7, 13, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        failure = SimpleNamespace(
            stage="discovery_fetch",
            error_type="http_status",
            source_url="https://example.test/list",
        )
        results = (
            _discovery_result(urls=()),
            _discovery_result(complete=False),
            _discovery_result(failures=(failure,)),
            _discovery_result(published_at=end),
        )

        for result in results:
            with self.subTest(result=result):
                with (
                    patch.object(
                        verify_news_e2e,
                        "_discover_pr_newswire",
                        new=AsyncMock(return_value=result),
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    passed, returned = await verify_news_e2e._check_discovery(
                        start, end
                    )
                self.assertFalse(passed)
                self.assertIsNone(returned)

    def test_ticker_hint_control_counts_markup_independently(self):
        documents = (
            _document(
                "Carnival (NYSE: [CCL](#financial-modal)) and "
                "NVIDIA (NASDAQ: NVDA).",
                "CCL",
                "NVDA",
            ),
            _document("Example (NASDAQ: **ABC**)."),
        )

        expected, recovered = verify_news_e2e._ticker_hint_control_counts(
            documents
        )

        self.assertEqual(expected, 3)
        self.assertEqual(recovered, 2)

    async def test_ticker_hint_check_enforces_full_sample_recall(self):
        discovery = _discovery_result()
        recovered = _document(
            "Carnival (NYSE: [CCL](#financial-modal)).",
            "CCL",
        )
        missed = _document("Example (NASDAQ: **ABC**).")
        no_ticker = _document("Example release with no ticker mention.")

        cases = (
            (recovered, True, "[PASS]"),
            (missed, False, "[FAIL]"),
            (no_ticker, True, "[SKIP]"),
        )
        for document, expected, state in cases:
            with self.subTest(expected=expected, state=state):
                with (
                    patch.object(
                        verify_news_e2e,
                        "_collect_observation",
                        new=AsyncMock(return_value=document),
                    ),
                    redirect_stdout(io.StringIO()) as output,
                ):
                    passed = await verify_news_e2e._check_ticker_hints(
                        discovery
                    )
                self.assertIs(passed, expected)
                self.assertIn(state, output.getvalue())


if __name__ == "__main__":
    unittest.main()
