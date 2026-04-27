"""Tests for configs.news."""

import unittest

from pydantic import TypeAdapter, ValidationError

from quantmind.configs.news import (
    Headline,
    HttpUrl,
    NewsFlowCfg,
    NewsInput,
    RssFeed,
)


class NewsFlowCfgTests(unittest.TestCase):
    def test_defaults(self):
        cfg = NewsFlowCfg()
        self.assertEqual(cfg.model, "gpt-4o")
        self.assertEqual(cfg.materiality_threshold, "medium")


class NewsInputTests(unittest.TestCase):
    def setUp(self):
        self.adapter = TypeAdapter(NewsInput)

    def test_rss(self):
        v = self.adapter.validate_python(
            {"type": "rss", "url": "https://feeds.example.com/markets"}
        )
        self.assertIsInstance(v, RssFeed)

    def test_http(self):
        v = self.adapter.validate_python(
            {"type": "http", "url": "https://news.example.com/a"}
        )
        self.assertIsInstance(v, HttpUrl)

    def test_headline(self):
        v = self.adapter.validate_python(
            {"type": "headline", "text": "Fed holds rates"}
        )
        self.assertIsInstance(v, Headline)

    def test_unknown_rejected(self):
        with self.assertRaises(ValidationError):
            self.adapter.validate_python({"type": "podcast", "url": "x"})


if __name__ == "__main__":
    unittest.main()
