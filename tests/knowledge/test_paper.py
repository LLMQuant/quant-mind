"""Tests for knowledge.paper."""

import unittest
from datetime import datetime, timezone

from quantmind.knowledge.paper import Paper


class PaperTests(unittest.TestCase):
    def test_minimal(self):
        p = Paper(
            as_of=datetime(2026, 4, 1, tzinfo=timezone.utc),
            summary="A momentum study.",
        )
        self.assertEqual(p.item_type, "paper")
        self.assertEqual(p.summary, "A momentum study.")
        self.assertEqual(p.key_findings, [])
        self.assertEqual(p.asset_classes, [])

    def test_full(self):
        p = Paper(
            as_of=datetime(2026, 4, 1, tzinfo=timezone.utc),
            summary="s",
            methodology="m",
            key_findings=["f1", "f2"],
            limitations=["l1"],
            asset_classes=["equities"],
            authors=["A. Smith"],
            arxiv_id="2604.12345",
        )
        self.assertEqual(p.arxiv_id, "2604.12345")
        self.assertEqual(p.asset_classes, ["equities"])


if __name__ == "__main__":
    unittest.main()
