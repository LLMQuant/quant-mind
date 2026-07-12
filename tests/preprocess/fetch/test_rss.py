"""Tests for preprocess.fetch.rss — RSS/Atom feed parsing."""

import unittest
from datetime import timezone

import httpx
import respx

from quantmind.preprocess.fetch.rss import fetch_rss_feed, parse_feed

_RSS = b"""\
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Press releases</title>
    <item>
      <title>NVIDIA Announces Results</title>
      <link>https://example.com/pr/nvidia-results?utm_source=x</link>
      <guid>gnw-123</guid>
      <pubDate>Mon, 15 Apr 2024 10:30:00 GMT</pubDate>
      <description><![CDATA[<p>Short summary</p>]]></description>
      <content:encoded><![CDATA[<article><p>Full release body</p></article>]]></content:encoded>
    </item>
  </channel>
</rss>
"""

_ATOM = b"""\
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom press releases</title>
  <entry>
    <title>Microsoft Announces Cloud Update</title>
    <id>tag:example.com,2024:msft-cloud</id>
    <updated>2024-04-15T10:30:00Z</updated>
    <link rel="alternate" href="https://example.com/pr/msft-cloud" />
    <summary type="html">&lt;p&gt;Cloud update summary&lt;/p&gt;</summary>
  </entry>
</feed>
"""


class ParseFeedTests(unittest.TestCase):
    def test_parse_rss_item_metadata(self):
        feed = parse_feed(
            _RSS,
            feed_url="https://example.com/rss",
            content_type="application/rss+xml",
            headers={"etag": "abc"},
        )

        self.assertEqual(feed.title, "Press releases")
        self.assertEqual(feed.feed_url, "https://example.com/rss")
        self.assertEqual(feed.content_type, "application/rss+xml")
        self.assertEqual(feed.headers["etag"], "abc")
        self.assertEqual(len(feed.items), 1)
        item = feed.items[0]
        self.assertEqual(item.title, "NVIDIA Announces Results")
        self.assertEqual(item.id, "gnw-123")
        self.assertEqual(
            item.url, "https://example.com/pr/nvidia-results?utm_source=x"
        )
        self.assertEqual(item.published_at.tzinfo, timezone.utc)
        self.assertEqual(item.published_at.hour, 10)
        self.assertIn("Full release body", item.content_html or "")

    def test_parse_atom_entry_metadata(self):
        feed = parse_feed(_ATOM, feed_url="https://example.com/atom")

        self.assertEqual(feed.title, "Atom press releases")
        self.assertEqual(len(feed.items), 1)
        item = feed.items[0]
        self.assertEqual(item.title, "Microsoft Announces Cloud Update")
        self.assertEqual(item.id, "tag:example.com,2024:msft-cloud")
        self.assertEqual(item.url, "https://example.com/pr/msft-cloud")
        self.assertEqual(item.published_at.tzinfo, timezone.utc)

    def test_unsupported_root_raises(self):
        with self.assertRaises(ValueError):
            parse_feed(b"<html></html>")


class FetchRssFeedTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_and_parses_feed(self):
        with respx.mock(assert_all_called=True) as router:
            router.get("https://example.com/rss").mock(
                return_value=httpx.Response(
                    200,
                    headers={"Content-Type": "application/rss+xml"},
                    content=_RSS,
                )
            )
            feed = await fetch_rss_feed("https://example.com/rss")

        self.assertEqual(feed.title, "Press releases")
        self.assertEqual(len(feed.items), 1)


if __name__ == "__main__":
    unittest.main()
