"""Tests for knowledge.earnings."""

import unittest
from datetime import datetime, timezone

from quantmind.knowledge.earnings import Earnings


class EarningsTests(unittest.TestCase):
    def test_minimal(self):
        e = Earnings(
            as_of=datetime(2026, 4, 26, tzinfo=timezone.utc),
            ticker="AAPL",
            period="2026Q1",
        )
        self.assertEqual(e.item_type, "earnings")
        self.assertIsNone(e.revenue)
        self.assertEqual(e.surprise_flags, [])

    def test_full(self):
        e = Earnings(
            as_of=datetime(2026, 4, 26, tzinfo=timezone.utc),
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


if __name__ == "__main__":
    unittest.main()
