"""Tests for provider-pluggable wire ingestion."""

import unittest
from pathlib import Path

import httpx
import respx

from quantmind.preprocess.fetch.http import FetchPolicy
from quantmind.preprocess.wire import (
    PR_NEWSWIRE,
    WireFeedConfig,
    WireItemMapping,
    build_wire_identity,
    fetch_wire_documents,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "wire"


def _fixture(name: str) -> bytes:
    return (_FIXTURES / name).read_bytes()


class _FeedBodyProvider:
    name = "full-content-test-wire"
    publisher = "Full Content Test Wire"
    body_source = "feed"

    def map_item(self, item):
        return WireItemMapping(
            payload_id=item.id,
            canonical_url=item.url,
            title=item.title,
            published_at=item.published_at,
        )


class WireProviderConfigTests(unittest.TestCase):
    def test_builtin_body_sources_are_explicit(self):
        self.assertEqual(PR_NEWSWIRE.body_source, "article")

    def test_config_requires_a_feed_url(self):
        with self.assertRaises(ValueError):
            WireFeedConfig(provider=PR_NEWSWIRE, feed_urls=())

    def test_identity_uses_all_available_components(self):
        base = build_wire_identity(
            provider="provider-a",
            payload_id="item-1",
            canonical_url="https://example.test/release",
        )

        self.assertNotEqual(
            base,
            build_wire_identity(
                provider="provider-b",
                payload_id="item-1",
                canonical_url="https://example.test/release",
            ),
        )
        self.assertNotEqual(
            base,
            build_wire_identity(
                provider="provider-a",
                payload_id="item-2",
                canonical_url="https://example.test/release",
            ),
        )
        self.assertNotEqual(
            base,
            build_wire_identity(
                provider="provider-a",
                payload_id="item-1",
                canonical_url="https://example.test/other",
            ),
        )


class FetchWireDocumentsTests(unittest.IsolatedAsyncioTestCase):
    async def test_feed_body_mode_remains_available_for_custom_provider(self):
        feed_url = "https://example.test/custom/rss"
        feed = b"""
        <rss><channel><item>
          <title>Example quarterly results</title>
          <link>https://example.test/releases/results</link>
          <guid>custom-example-001</guid>
          <description><![CDATA[
            <p>Example Corp reported fictional quarterly revenue.</p>
            <p>NASDAQ: EXMPL</p>
          ]]></description>
        </item></channel></rss>
        """
        with respx.mock(assert_all_called=True) as router:
            router.get(feed_url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "application/rss+xml"},
                    content=feed,
                )
            )
            result = await fetch_wire_documents(
                WireFeedConfig(
                    provider=_FeedBodyProvider(),
                    feed_urls=(feed_url,),
                )
            )

        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.failure_count, 0)
        document = result.documents[0]
        self.assertIn("fictional quarterly revenue", document.cleaned_markdown)
        self.assertIsNone(document.raw_article)
        self.assertIn(b"custom-example-001", document.raw_feed_entry.bytes)
        self.assertEqual(document.raw_feed_entry.status_code, 200)
        self.assertIsNotNone(document.raw_feed_entry.fetched_at)
        self.assertEqual(document.ticker_hints[0].symbol, "EXMPL")

    async def test_pr_newswire_returns_both_raw_artifacts(self):
        feed_url = "https://example.test/prn/rss"
        article_url = "https://example.test/prn/releases/operations-update"
        article = _fixture("pr_news_wire_article.html")
        with respx.mock(assert_all_called=True) as router:
            router.get(feed_url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "application/rss+xml"},
                    content=_fixture("pr_news_wire.xml"),
                )
            )
            router.get(article_url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    content=article,
                )
            )
            result = await fetch_wire_documents(
                WireFeedConfig(
                    provider=PR_NEWSWIRE,
                    feed_urls=(feed_url,),
                )
            )

        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.failure_count, 0)
        document = result.documents[0]
        self.assertIn("full article", document.cleaned_markdown)
        self.assertIsNotNone(document.raw_article)
        assert document.raw_article is not None
        self.assertEqual(document.raw_article.bytes, article)
        self.assertTrue(document.raw_feed_entry.bytes)

    async def test_duplicate_identity_is_processed_once(self):
        feed_url = "https://example.test/duplicate/rss"
        item = """
        <item>
          <title>Duplicate release</title>
          <link>https://example.test/releases/duplicate</link>
          <guid>duplicate-1</guid>
          <description><![CDATA[<p>A complete duplicate body.</p>]]></description>
        </item>
        """
        feed = f"<rss><channel>{item}{item}</channel></rss>".encode()
        with respx.mock(assert_all_called=True) as router:
            router.get(feed_url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "application/rss+xml"},
                    content=feed,
                )
            )
            result = await fetch_wire_documents(
                WireFeedConfig(
                    provider=_FeedBodyProvider(),
                    feed_urls=(feed_url,),
                )
            )

        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.failure_count, 0)

    async def test_extraction_failure_is_recorded_and_batch_continues(self):
        feed_url = "https://example.test/partial/rss"
        bad_url = "https://example.test/releases/bad"
        good_url = "https://example.test/releases/good"
        feed = f"""
        <rss><channel>
          <item>
            <title>Bad release</title><link>{bad_url}</link>
            <guid>bad-1</guid><description>Short teaser</description>
          </item>
          <item>
            <title>Good release</title><link>{good_url}</link>
            <guid>good-1</guid><description>Short teaser</description>
          </item>
        </channel></rss>
        """.encode()
        good_article = b"""
        <html><body><article>
          <h1>Good release</h1><p>The complete good article body.</p>
        </article></body></html>
        """
        with respx.mock(assert_all_called=True) as router:
            router.get(feed_url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "application/rss+xml"},
                    content=feed,
                )
            )
            router.get(bad_url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "application/pdf"},
                    content=b"not html",
                )
            )
            router.get(good_url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    content=good_article,
                )
            )
            result = await fetch_wire_documents(
                WireFeedConfig(
                    provider=PR_NEWSWIRE,
                    feed_urls=(feed_url,),
                    fetch_policy=FetchPolicy(max_attempts=1),
                )
            )

        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.failure_count, 1)
        self.assertEqual(result.documents[0].payload_id, "good-1")
        failure = result.failures[0]
        self.assertEqual(failure.item_id, "bad-1")
        self.assertEqual(failure.stage, "article_parse")
        self.assertEqual(failure.error_type, "invalid_content")

    async def test_feed_failure_does_not_discard_other_feed(self):
        bad_feed = "https://example.test/feed/bad"
        good_feed = "https://example.test/feed/good"
        with respx.mock(assert_all_called=True) as router:
            router.get(bad_feed).mock(return_value=httpx.Response(404))
            router.get(good_feed).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "application/rss+xml"},
                    content=b"""
                    <rss><channel><item>
                      <title>Good feed item</title>
                      <link>https://example.test/releases/good-feed</link>
                      <guid>good-feed-1</guid>
                      <description>A complete feed body.</description>
                    </item></channel></rss>
                    """,
                )
            )
            result = await fetch_wire_documents(
                WireFeedConfig(
                    provider=_FeedBodyProvider(),
                    feed_urls=(bad_feed, good_feed),
                    fetch_policy=FetchPolicy(max_attempts=1),
                )
            )

        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.failure_count, 1)
        self.assertEqual(result.failures[0].stage, "feed_fetch")
        self.assertEqual(result.failures[0].error_type, "http_status")


if __name__ == "__main__":
    unittest.main()
