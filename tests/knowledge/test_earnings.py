"""Tests for knowledge.earnings."""

import unittest
from datetime import datetime, timezone

from quantmind.knowledge._base import SourceRef
from quantmind.knowledge.earnings import Earnings


def _now() -> datetime:
    return datetime(2026, 4, 26, tzinfo=timezone.utc)


def _src() -> SourceRef:
    return SourceRef(kind="http", uri="https://ir.example.com/q1.pdf")


class EarningsTests(unittest.TestCase):
    def test_minimal(self):
        e = Earnings(
            as_of=_now(),
            source=_src(),
            ticker="AAPL",
            period="2026Q1",
        )
        self.assertEqual(e.item_type, "earnings")
        self.assertIsNone(e.revenue)
        self.assertEqual(e.surprise_flags, [])

    def test_full(self):
        e = Earnings(
            as_of=_now(),
            source=_src(),
            ticker="AAPL",
            period="2026Q1",
            revenue=120.0,
            eps=1.55,
            guidance="Raised FY revenue guide",
            surprise_flags=["eps_beat", "revenue_beat"],
            transcript_quote="Demand remains robust ...",
        )
        self.assertEqual(e.revenue, 120.0)
        self.assertEqual(e.surprise_flags, ["eps_beat", "revenue_beat"])

    def test_embedding_text(self):
        e = Earnings(
            as_of=_now(),
            source=_src(),
            ticker="AAPL",
            period="2026Q1",
            guidance="Raised FY guide",
        )
        self.assertEqual(
            e.embedding_text(),
            "AAPL 2026Q1 earnings\nRaised FY guide",
        )


if __name__ == "__main__":
    unittest.main()
