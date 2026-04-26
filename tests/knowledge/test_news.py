"""Tests for knowledge.news."""

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from quantmind.knowledge.news import News


class NewsTests(unittest.TestCase):
    def test_minimal(self):
        n = News(
            as_of=datetime(2026, 4, 26, tzinfo=timezone.utc),
            headline="Fed holds rates",
            event_type="monetary_policy",
            timestamp=datetime(2026, 4, 26, tzinfo=timezone.utc),
        )
        self.assertEqual(n.item_type, "news")
        self.assertEqual(n.sentiment, "neutral")
        self.assertEqual(n.materiality, "medium")

    def test_sentiment_enum(self):
        with self.assertRaises(ValidationError):
            News(
                as_of=datetime(2026, 4, 26, tzinfo=timezone.utc),
                headline="x",
                event_type="x",
                timestamp=datetime(2026, 4, 26, tzinfo=timezone.utc),
                sentiment="ecstatic",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
