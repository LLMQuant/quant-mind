"""Tests for knowledge._base."""

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from quantmind.knowledge._base import Citation, KnowledgeItem


class CitationTests(unittest.TestCase):
    def test_minimal(self):
        cit = Citation(source_id="arxiv:2604.12345")
        self.assertEqual(cit.source_id, "arxiv:2604.12345")
        self.assertIsNone(cit.page)
        self.assertIsNone(cit.quote)

    def test_quote_max_length(self):
        with self.assertRaises(ValidationError):
            Citation(source_id="x", quote="a" * 501)


class KnowledgeItemTests(unittest.TestCase):
    def _now(self) -> datetime:
        return datetime(2026, 4, 26, tzinfo=timezone.utc)

    def test_as_of_required(self):
        with self.assertRaises(ValidationError):
            KnowledgeItem(item_type="generic")  # type: ignore[call-arg]

    def test_default_confidence_is_medium(self):
        item = KnowledgeItem(item_type="generic", as_of=self._now())
        self.assertEqual(item.confidence, "medium")

    def test_frozen(self):
        item = KnowledgeItem(item_type="generic", as_of=self._now())
        with self.assertRaises(ValidationError):
            item.tags = ["new"]  # type: ignore[misc]

    def test_extra_forbidden(self):
        with self.assertRaises(ValidationError):
            KnowledgeItem(
                item_type="generic",
                as_of=self._now(),
                unexpected_field=1,  # type: ignore[call-arg]
            )


class PackageExportTests(unittest.TestCase):
    def test_top_level_imports(self):
        from quantmind.knowledge import (
            Citation,
            Earnings,
            KnowledgeItem,
            News,
            Paper,
        )

        self.assertTrue(issubclass(Paper, KnowledgeItem))
        self.assertTrue(issubclass(News, KnowledgeItem))
        self.assertTrue(issubclass(Earnings, KnowledgeItem))
        self.assertEqual(Citation.__name__, "Citation")


if __name__ == "__main__":
    unittest.main()
