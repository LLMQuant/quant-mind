"""Tests for preprocess.format.html — html_to_markdown via trafilatura."""

import unittest
from pathlib import Path

from quantmind.preprocess.format.html import html_to_markdown

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample.html"


class HtmlToMarkdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_extracts_article_body(self):
        html = _FIXTURE.read_text(encoding="utf-8")
        markdown = await html_to_markdown(html)
        self.assertTrue(markdown.strip(), "expected non-empty markdown output")
        # Strict-precision mode should keep the article paragraphs and drop
        # the nav / footer chrome.
        self.assertIn("trafilatura", markdown)

    async def test_strips_boilerplate_by_default(self):
        html = _FIXTURE.read_text(encoding="utf-8")
        markdown = await html_to_markdown(html)
        # Footer copyright string should be gone in strict mode.
        self.assertNotIn("Cookie policy", markdown)

    async def test_returns_string(self):
        html = _FIXTURE.read_text(encoding="utf-8")
        markdown = await html_to_markdown(html)
        self.assertIsInstance(markdown, str)

    async def test_empty_html_returns_empty_string(self):
        result = await html_to_markdown("   ")
        self.assertEqual(result, "")

    async def test_unparseable_returns_empty_string(self):
        # A bare snippet with no extractable article body — trafilatura
        # returns None which our wrapper translates to "".
        result = await html_to_markdown("<html><body></body></html>")
        self.assertEqual(result, "")
