"""Tests for preprocess.fetch._types — Fetched + RawPaper invariants."""

import unittest
from datetime import datetime, timezone

from quantmind.preprocess.fetch._types import Fetched, RawPaper


class FetchedTests(unittest.TestCase):
    def test_minimal_construction(self):
        f = Fetched(bytes=b"hello", content_type="text/plain")
        self.assertEqual(f.bytes, b"hello")
        self.assertEqual(f.content_type, "text/plain")
        self.assertIsNone(f.source_url)
        self.assertEqual(f.headers, {})

    def test_frozen(self):
        f = Fetched(bytes=b"x", content_type="text/plain")
        with self.assertRaises(Exception):
            f.bytes = b"y"  # type: ignore[misc]

    def test_default_headers_are_independent(self):
        a = Fetched(bytes=b"", content_type="x")
        b = Fetched(bytes=b"", content_type="y")
        self.assertIsNot(a.headers, b.headers)


class RawPaperTests(unittest.TestCase):
    def test_inherits_fetched_fields(self):
        p = RawPaper(
            bytes=b"%PDF",
            content_type="application/pdf",
            source_url="https://arxiv.org/pdf/2401.12345",
            headers={},
            arxiv_id="2401.12345",
            title="A Test Paper",
            authors=("Alice", "Bob"),
            abstract="abstract text",
            published_at=datetime(2024, 4, 15, tzinfo=timezone.utc),
            primary_category="q-fin.ST",
            categories=("q-fin.ST", "stat.ML"),
        )
        self.assertEqual(p.arxiv_id, "2401.12345")
        self.assertEqual(p.authors, ("Alice", "Bob"))
        self.assertEqual(p.bytes, b"%PDF")

    def test_authors_stored_as_tuple(self):
        # Tuple is required for hashability of frozen dataclass.
        p = RawPaper(bytes=b"", content_type="application/pdf")
        self.assertIsInstance(p.authors, tuple)
        self.assertIsInstance(p.categories, tuple)

    def test_defaults_present(self):
        p = RawPaper(bytes=b"", content_type="application/pdf")
        self.assertEqual(p.arxiv_id, "")
        self.assertIsNone(p.title)
        self.assertEqual(p.authors, ())
        self.assertIsNone(p.published_at)
