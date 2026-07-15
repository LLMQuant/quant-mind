"""Tests for preprocess.news — PR/wire candidate normalisation."""

import hashlib
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import respx

from quantmind.preprocess.fetch.rss import FeedItem
from quantmind.preprocess.news import (
    RawNewsDocument,
    build_news_identity,
    build_sec_news_identity,
    canonicalize_source_url,
    extract_exchange_ticker_hints,
    feed_item_to_news_document,
    fetch_news_document,
    news_content_hash,
    normalize_news_text,
    preprocess_feed_item,
    preprocess_news_document,
)
from quantmind.preprocess.time import parse_news_datetime

_FIXTURES = Path(__file__).parent / "fixtures" / "pr_newswire"


class NewsTimeTests(unittest.TestCase):
    def test_parse_rfc822_news_datetime(self):
        result = parse_news_datetime("Mon, 15 Apr 2024 10:30:00 GMT")

        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.hour, 10)

    def test_parse_news_datetime_with_offset(self):
        result = parse_news_datetime("Mon, 15 Apr 2024 10:30:00 -0400")

        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.hour, 14)


class NewsPreprocessTests(unittest.TestCase):
    def test_normalize_news_text_and_hash(self):
        text = (
            "NVIDIA\u2014reported   revenue\n\n\nNVIDIA\u2014reported   revenue"
        )
        normalized = normalize_news_text(text)

        self.assertEqual(
            normalized,
            "NVIDIA-reported revenue\n\nNVIDIA-reported revenue",
        )
        self.assertEqual(
            news_content_hash(normalized),
            hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        )

    def test_normalize_news_text_stabilizes_email_protection_links(self):
        text_a = (
            "Contact [[email protected]](/cdn-cgi/l/email-protection#abc123)"
        )
        text_b = (
            "Contact [[email protected]](/cdn-cgi/l/email-protection#def456)"
        )

        self.assertEqual(
            normalize_news_text(text_a), normalize_news_text(text_b)
        )

    def test_canonicalize_source_url_drops_tracking(self):
        result = canonicalize_source_url(
            "HTTPS://Example.COM/path/?b=2&utm_source=x&a=1#frag"
        )

        self.assertEqual(result, "https://example.com/path?a=1&b=2")

    def test_exchange_ticker_hints_are_deduped(self):
        hints = extract_exchange_ticker_hints(
            "NVIDIA Corporation (NASDAQ: NVDA) and IBM NYSE: IBM. "
            "Again (NASDAQ: NVDA)."
        )

        self.assertEqual([h.symbol for h in hints], ["NVDA", "IBM"])
        self.assertEqual(hints[0].exchange, "NASDAQ")
        self.assertEqual(hints[1].exchange, "NYSE")

    def test_exchange_ticker_hints_ignore_markdown_decoration(self):
        cases = (
            (
                "link",
                "Carnival Corporation & plc "
                "(NYSE: [CCL](#financial-modal)) today announced...",
                ("CCL", "NYSE", "(NYSE: CCL)"),
            ),
            (
                "bold emphasis",
                "Example Corporation (NASDAQ: **ABC**) today announced...",
                ("ABC", "NASDAQ", "(NASDAQ: ABC)"),
            ),
            (
                "italic emphasis",
                "Example Corporation (NASDAQ: *ABC*) today announced...",
                ("ABC", "NASDAQ", "(NASDAQ: ABC)"),
            ),
        )

        for name, text, expected in cases:
            with self.subTest(name=name):
                hints = extract_exchange_ticker_hints(text)

                self.assertEqual(len(hints), 1)
                self.assertEqual(
                    (hints[0].symbol, hints[0].exchange, hints[0].raw),
                    expected,
                )

    def test_build_sec_news_identity(self):
        self.assertEqual(
            build_sec_news_identity(
                accession_number="0001045810-26-000123",
                section_key="EX99.1",
            ),
            "sec:0001045810-26-000123:ex99.1",
        )

    def test_build_news_identity_requires_source_reference(self):
        with self.assertRaises(ValueError):
            build_news_identity(source_type="press_release")

    def test_preprocess_news_document_builds_candidate_contract(self):
        published = datetime(
            2024, 4, 15, 10, 30, tzinfo=timezone(timedelta(hours=-4))
        )
        raw = RawNewsDocument(
            body_text=(
                "NVIDIA Corporation (NASDAQ: NVDA) today reported "
                "record quarterly revenue."
            ),
            source_url="https://example.com/pr/nvidia-results?utm_source=feed",
            title="NVIDIA Announces Results",
            publisher="PR Newswire",
            published_at=published,
            payload_id="gnw-123",
        )

        candidate = preprocess_news_document(raw)

        self.assertEqual(candidate.source_type, "press_release")
        self.assertTrue(candidate.identity.startswith("wire:"))
        self.assertEqual(
            candidate.source_url, "https://example.com/pr/nvidia-results"
        )
        self.assertEqual(candidate.published_at.tzinfo, timezone.utc)
        self.assertEqual(candidate.published_at.hour, 14)
        self.assertEqual(candidate.ticker_hints[0].symbol, "NVDA")
        self.assertEqual(
            candidate.content_hash,
            hashlib.sha256(candidate.body_text.encode("utf-8")).hexdigest(),
        )

    def test_preprocess_news_document_rejects_empty_body(self):
        raw = RawNewsDocument(
            body_text="  ",
            source_url="https://example.com/pr/empty",
        )

        with self.assertRaises(ValueError):
            preprocess_news_document(raw)

    def test_wire_markup_fixture_meets_ticker_recall_control(self):
        body_text = (_FIXTURES / "ticker_markup.md").read_text(encoding="utf-8")
        raw = RawNewsDocument(
            body_text=body_text,
            source_url="https://www.prnewswire.com/news-releases/example.html",
        )

        candidate = preprocess_news_document(raw)

        expected_hints = {("ABC", "NASDAQ"), ("CCL", "NYSE")}
        actual_hints = {
            (hint.symbol, hint.exchange) for hint in candidate.ticker_hints
        }
        recall = len(expected_hints & actual_hints) / len(expected_hints)
        self.assertEqual(recall, 1.0)
        self.assertIn("[CCL](#financial-modal)", candidate.body_text)
        self.assertIn("**ABC**", candidate.body_text)
        self.assertEqual(
            candidate.content_hash,
            news_content_hash(candidate.body_text),
        )


class NewsFeedItemTests(unittest.IsolatedAsyncioTestCase):
    async def test_feed_item_to_news_document_uses_inline_content(self):
        item = FeedItem(
            title="NVIDIA Announces Results",
            url="https://example.com/pr/nvidia-results",
            id="gnw-123",
            published_at=datetime(2024, 4, 15, tzinfo=timezone.utc),
            content_html=(
                "<html><body><article><p>NVIDIA Corporation "
                "(NASDAQ: NVDA) reported results.</p></article></body></html>"
            ),
            source_feed_url="https://example.com/rss",
        )

        raw = await feed_item_to_news_document(item, publisher="PR Newswire")

        self.assertEqual(raw.payload_id, "gnw-123")
        self.assertEqual(raw.publisher, "PR Newswire")
        self.assertIn("NVIDIA", raw.body_text)
        self.assertEqual(
            raw.metadata["source_feed_url"], "https://example.com/rss"
        )
        self.assertEqual(raw.metadata["body_source"], "feed")

    async def test_article_source_fetches_despite_nonempty_teaser(self):
        item = FeedItem(
            title="NVIDIA Announces Results",
            url="https://example.com/pr/nvidia-results",
            id="wire-123",
            summary_html="<p>This non-empty teaser is not the full body.</p>",
            source_feed_url="https://example.com/rss",
        )
        html = """
        <html><body><article>
          <h1>NVIDIA Announces Results</h1>
          <p>The complete release contains full financial details.</p>
        </article></body></html>
        """
        with respx.mock(assert_all_called=True) as router:
            router.get(item.url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    text=html,
                )
            )
            raw = await feed_item_to_news_document(
                item,
                body_source="article",
            )

        self.assertIn("complete release", raw.body_text)
        self.assertNotIn("non-empty teaser", raw.body_text)
        self.assertEqual(raw.metadata["body_source"], "article")

    async def test_feed_source_does_not_fetch_item_url(self):
        item = FeedItem(
            title="NVIDIA Announces Results",
            url="https://example.com/pr/nvidia-results",
            summary_html="<p>Feed body stays local.</p>",
        )

        with respx.mock(assert_all_called=True):
            raw = await feed_item_to_news_document(
                item,
                body_source="feed",
            )

        self.assertIn("Feed body", raw.body_text)

    async def test_article_source_requires_item_url(self):
        item = FeedItem(title="Missing URL", summary_html="A teaser")

        with self.assertRaisesRegex(ValueError, "requires an item URL"):
            await feed_item_to_news_document(
                item,
                body_source="article",
            )

    async def test_article_extraction_failure_is_explicit(self):
        item = FeedItem(
            title="Unsupported body",
            url="https://example.com/release.pdf",
            summary_html="A teaser",
        )
        with respx.mock(assert_all_called=True) as router:
            router.get(item.url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "application/pdf"},
                    content=b"not html",
                )
            )
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                await feed_item_to_news_document(
                    item,
                    body_source="article",
                )

    async def test_preprocess_feed_item_returns_candidate(self):
        item = FeedItem(
            title="NVIDIA Announces Results",
            url="https://example.com/pr/nvidia-results",
            id="gnw-123",
            summary_html="<p>NVIDIA Corporation (NASDAQ: NVDA) reported results.</p>",
        )

        candidate = await preprocess_feed_item(item, publisher="PR Newswire")

        self.assertEqual(candidate.publisher, "PR Newswire")
        self.assertEqual(candidate.ticker_hints[0].symbol, "NVDA")


class FetchNewsDocumentTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_news_document_extracts_html(self):
        html = """
        <html>
          <body>
            <article>
              <h1>NVIDIA Announces Results</h1>
              <p>NVIDIA Corporation (NASDAQ: NVDA) reported record revenue.</p>
            </article>
          </body>
        </html>
        """
        with respx.mock(assert_all_called=True) as router:
            router.get("https://example.com/pr/nvidia-results").mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "text/html; charset=utf-8"},
                    content=html.encode(),
                )
            )
            raw = await fetch_news_document(
                "https://example.com/pr/nvidia-results",
                title="NVIDIA Announces Results",
                payload_id="gnw-123",
            )

        self.assertEqual(
            raw.source_url, "https://example.com/pr/nvidia-results"
        )
        self.assertEqual(raw.metadata["content_type"], "text/html")
        self.assertIn("record revenue", raw.body_text)


if __name__ == "__main__":
    unittest.main()
