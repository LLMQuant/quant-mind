import re
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
import respx

from quantmind.preprocess.fetch.http import FetchPolicy
from quantmind.preprocess.pr_newswire import (
    _collect_pr_newswire,
    _discover_pr_newswire,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "pr_newswire"
_LISTING_RE = re.compile(
    r"https://www\.prnewswire\.com/news-releases/"
    r"news-releases-list/\?.*"
)
_TEST_POLICY = FetchPolicy(
    max_attempts=1,
    backoff_base_seconds=0.0,
    backoff_max_seconds=0.0,
    jitter_seconds=0.0,
    max_concurrency_per_host=2,
    min_interval_seconds=0.0,
)


def _fixture(name: str) -> bytes:
    return (_FIXTURES / name).read_bytes()


def _listing_response(content: bytes, request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        headers={"Content-Type": "text/html"},
        content=content,
    )


def _short_listing(*rows: tuple[str, str]) -> bytes:
    cards = "".join(
        (
            '<div class="row newsCards"><div>'
            '<a class="newsreleaseconsolidatelink" '
            f'href="/news-releases/example-{payload_id}.html">'
            f"<h3><small>{timestamp}</small>Example {payload_id}</h3>"
            "</a></div></div>"
        )
        for timestamp, payload_id in rows
    )
    return (
        '<html><body><input id="date" value="07/14/2026" />'
        f"{cards}</body></html>"
    ).encode()


class DiscoverPRNewswireTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_duplicates_and_rolls_short_times_across_pages(
        self,
    ) -> None:
        pages = {
            1: _fixture("listing_short_page_1.html"),
            2: _fixture("listing_short_page_2.html"),
        }

        def respond(request: httpx.Request) -> httpx.Response:
            page = int(request.url.params["page"])
            return _listing_response(pages[page], request)

        with (
            patch(
                "quantmind.preprocess.pr_newswire._DEFAULT_FETCH_POLICY",
                _TEST_POLICY,
            ),
            respx.mock(assert_all_called=True) as router,
        ):
            route = router.get(url__regex=_LISTING_RE).mock(side_effect=respond)
            result = await _discover_pr_newswire(
                start=datetime(2026, 7, 14, 3, 58, tzinfo=timezone.utc),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
            )

        self.assertTrue(result.complete)
        self.assertEqual(result.page_count, 2)
        self.assertEqual(result.observed_count, 4)
        self.assertEqual(route.call_count, 2)
        self.assertEqual(
            [item.payload_id for item in result.observations],
            ["302900111", "302900111", "302900112", "302900113"],
        )
        self.assertEqual(
            result.observations[-1].published_at,
            datetime(2026, 7, 14, 3, 59, tzinfo=timezone.utc),
        )
        self.assertEqual(
            result.observations[0].identity,
            result.observations[1].identity,
        )
        self.assertNotEqual(
            result.observations[0].canonical_url,
            result.observations[1].canonical_url,
        )
        self.assertEqual(
            len({item.identity for item in result.observations}), 3
        )
        artifact = result.observations[0].discovery_artifact
        self.assertIn(
            b"Repeated", result.observations[1].discovery_artifact.bytes
        )
        self.assertIn(b"Example Alpha", artifact.bytes)
        self.assertEqual(artifact.content_type, "text/html")
        self.assertEqual(artifact.status_code, 200)
        request = route.calls[0].request
        self.assertEqual(request.url.params["hour"], "01")

    async def test_full_timestamps_use_half_open_window_boundaries(
        self,
    ) -> None:
        with (
            patch(
                "quantmind.preprocess.pr_newswire._DEFAULT_FETCH_POLICY",
                _TEST_POLICY,
            ),
            respx.mock(assert_all_called=True) as router,
        ):
            router.get(url__regex=_LISTING_RE).mock(
                side_effect=lambda request: _listing_response(
                    _fixture("listing_full.html"), request
                )
            )
            result = await _discover_pr_newswire(
                start=datetime(2026, 7, 14, 3, 58, tzinfo=timezone.utc),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
            )

        self.assertTrue(result.complete)
        self.assertEqual(
            [item.payload_id for item in result.observations],
            ["302900202", "302900203"],
        )
        self.assertEqual(
            result.observations[-1].published_at,
            datetime(2026, 7, 14, 3, 58, tzinfo=timezone.utc),
        )

    async def test_equal_start_cluster_continues_onto_the_next_page(
        self,
    ) -> None:
        pages = {
            1: _short_listing(
                ("00:10 ET", "302900401"),
                ("00:00 ET", "302900402"),
            ),
            2: _short_listing(
                ("00:00 ET", "302900403"),
                ("23:59 ET", "302900404"),
            ),
        }

        def respond(request: httpx.Request) -> httpx.Response:
            return _listing_response(
                pages[int(request.url.params["page"])],
                request,
            )

        with (
            patch(
                "quantmind.preprocess.pr_newswire._DEFAULT_FETCH_POLICY",
                _TEST_POLICY,
            ),
            respx.mock(assert_all_called=True) as router,
        ):
            router.get(url__regex=_LISTING_RE).mock(side_effect=respond)
            result = await _discover_pr_newswire(
                start=datetime(2026, 7, 14, 4, 0, tzinfo=timezone.utc),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
            )

        self.assertTrue(result.complete)
        self.assertEqual(result.page_count, 2)
        self.assertEqual(
            [item.payload_id for item in result.observations],
            ["302900401", "302900402", "302900403"],
        )

    async def test_row_parse_failure_keeps_good_rows_but_is_incomplete(
        self,
    ) -> None:
        listing = b"""
        <html><body><input id="date" value="07/14/2026" />
          <div class="row newsCards"><div>
            <a class="newsreleaseconsolidatelink"
               href="/news-releases/good-302900301.html">
              <h3><small>Jul 14, 2026, 00:10 ET</small>Good row</h3>
            </a>
          </div></div>
          <div class="row newsCards"><div>
            <a class="newsreleaseconsolidatelink"
               href="/news-releases/bad-302900302.html">
              <h3><small>not-a-time</small>Bad row</h3>
            </a>
          </div></div>
          <div class="row newsCards"><div>
            <a class="newsreleaseconsolidatelink"
               href="/news-releases/older-302900303.html">
              <h3><small>Jul 13, 2026, 23:57 ET</small>Older row</h3>
            </a>
          </div></div>
        </body></html>
        """
        with (
            patch(
                "quantmind.preprocess.pr_newswire._DEFAULT_FETCH_POLICY",
                _TEST_POLICY,
            ),
            respx.mock(assert_all_called=True) as router,
        ):
            router.get(url__regex=_LISTING_RE).mock(
                side_effect=lambda request: _listing_response(listing, request)
            )
            result = await _discover_pr_newswire(
                start=datetime(2026, 7, 14, 3, 58, tzinfo=timezone.utc),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
            )

        self.assertFalse(result.complete)
        self.assertEqual(result.observed_count, 1)
        self.assertEqual(result.failures[0].stage, "discovery_parse")
        self.assertEqual(result.failures[0].error_type, "invalid_content")

    async def test_fetch_failure_is_recorded(self) -> None:
        with (
            patch(
                "quantmind.preprocess.pr_newswire._DEFAULT_FETCH_POLICY",
                _TEST_POLICY,
            ),
            respx.mock(assert_all_called=True) as router,
        ):
            router.get(url__regex=_LISTING_RE).mock(
                return_value=httpx.Response(400)
            )
            result = await _discover_pr_newswire(
                start=datetime(2026, 7, 14, 3, 58, tzinfo=timezone.utc),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
            )

        self.assertFalse(result.complete)
        self.assertEqual(result.page_count, 0)
        self.assertEqual(result.failures[0].stage, "discovery_fetch")
        self.assertEqual(result.failures[0].error_type, "http_status")

    async def test_page_limit_is_an_explicit_incomplete_failure(self) -> None:
        pages = {
            1: _fixture("listing_short_page_1.html"),
            2: _fixture("listing_short_page_2.html"),
        }

        def respond(request: httpx.Request) -> httpx.Response:
            return _listing_response(
                pages[int(request.url.params["page"])],
                request,
            )

        with (
            patch(
                "quantmind.preprocess.pr_newswire._DEFAULT_FETCH_POLICY",
                _TEST_POLICY,
            ),
            patch("quantmind.preprocess.pr_newswire._MAX_PAGES", 2),
            respx.mock(assert_all_called=True) as router,
        ):
            router.get(url__regex=_LISTING_RE).mock(side_effect=respond)
            result = await _discover_pr_newswire(
                start=datetime(2026, 7, 12, tzinfo=timezone.utc),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
            )

        self.assertFalse(result.complete)
        self.assertEqual(result.page_count, 2)
        self.assertIn("exceeded 2 pages", result.failures[0].message)

    async def test_cache_bust_retries_listing_404(self) -> None:
        with (
            patch(
                "quantmind.preprocess.pr_newswire._DEFAULT_FETCH_POLICY",
                _TEST_POLICY,
            ),
            respx.mock(assert_all_called=True) as router,
        ):
            route = router.get(url__regex=_LISTING_RE).mock(
                side_effect=[
                    httpx.Response(404),
                    httpx.Response(
                        200,
                        headers={"Content-Type": "text/html"},
                        content=_fixture("listing_full.html"),
                    ),
                ]
            )
            result = await _discover_pr_newswire(
                start=datetime(2026, 7, 14, 4, 29, tzinfo=timezone.utc),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
            )

        self.assertTrue(result.complete)
        self.assertEqual(route.call_count, 2)
        self.assertNotEqual(
            route.calls[0].request.url.params["_"],
            route.calls[1].request.url.params["_"],
        )

    async def test_rejects_invalid_windows_before_network_work(self) -> None:
        with self.assertRaisesRegex(ValueError, "aware start"):
            await _discover_pr_newswire(
                start=datetime(2026, 7, 14, 3, 58),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
            )
        with self.assertRaisesRegex(ValueError, "end after start"):
            await _discover_pr_newswire(
                start=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
            )


class CollectPRNewswireTests(unittest.IsolatedAsyncioTestCase):
    async def test_collects_article_and_discards_raw_html_by_default(
        self,
    ) -> None:
        article_url = (
            "https://www.prnewswire.com/news-releases/in-window-302900202.html"
        )
        with (
            patch(
                "quantmind.preprocess.pr_newswire._DEFAULT_FETCH_POLICY",
                _TEST_POLICY,
            ),
            respx.mock(assert_all_called=True) as router,
        ):
            router.get(url__regex=_LISTING_RE).mock(
                side_effect=lambda request: _listing_response(
                    _fixture("listing_full.html"), request
                )
            )
            router.get(article_url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    content=_fixture("article.html"),
                )
            )
            result = await _collect_pr_newswire(
                start=datetime(2026, 7, 14, 4, 29, tzinfo=timezone.utc),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
                retain_raw_html=False,
            )

        self.assertTrue(result.complete)
        self.assertEqual(result.observed_count, 1)
        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.failure_count, 0)
        document = result.documents[0]
        self.assertIn("fictional operating details", document.cleaned_markdown)
        self.assertIsNone(document.article_artifact.bytes)
        self.assertTrue(document.article_artifact.content_hash)
        self.assertTrue(document.discovery_artifact.bytes)
        self.assertEqual(
            document.identity.split(":", 2)[:2], ["news", "pr-newswire"]
        )
        self.assertEqual(document.ticker_hints[0].symbol, "EXCO")

    async def test_retain_raw_html_and_record_independent_article_failure(
        self,
    ) -> None:
        article_url = (
            "https://www.prnewswire.com/news-releases/in-window-302900202.html"
        )
        with (
            patch(
                "quantmind.preprocess.pr_newswire._DEFAULT_FETCH_POLICY",
                _TEST_POLICY,
            ),
            respx.mock(assert_all_called=True) as router,
        ):
            router.get(url__regex=_LISTING_RE).mock(
                side_effect=lambda request: _listing_response(
                    _fixture("listing_full.html"), request
                )
            )
            router.get(article_url).mock(return_value=httpx.Response(404))
            failed = await _collect_pr_newswire(
                start=datetime(2026, 7, 14, 4, 29, tzinfo=timezone.utc),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
                retain_raw_html=True,
            )

        self.assertTrue(failed.complete)
        self.assertEqual(failed.observed_count, 1)
        self.assertEqual(failed.success_count, 0)
        self.assertEqual(failed.failures[0].stage, "article_fetch")

        with (
            patch(
                "quantmind.preprocess.pr_newswire._DEFAULT_FETCH_POLICY",
                _TEST_POLICY,
            ),
            respx.mock(assert_all_called=True) as router,
        ):
            router.get(url__regex=_LISTING_RE).mock(
                side_effect=lambda request: _listing_response(
                    _fixture("listing_full.html"), request
                )
            )
            router.get(article_url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    content=_fixture("article.html"),
                )
            )
            retained = await _collect_pr_newswire(
                start=datetime(2026, 7, 14, 4, 29, tzinfo=timezone.utc),
                end=datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc),
                retain_raw_html=True,
            )

        self.assertEqual(
            retained.documents[0].article_artifact.bytes,
            _fixture("article.html"),
        )


if __name__ == "__main__":
    unittest.main()
