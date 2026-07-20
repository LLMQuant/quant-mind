"""Tests for knowledge.news."""

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from quantmind.knowledge._base import SourceRef
from quantmind.knowledge.news import News


def _now() -> datetime:
    return datetime(2026, 4, 26, tzinfo=timezone.utc)


def _src() -> SourceRef:
    return SourceRef(kind="rss", uri="https://feeds.example.com/markets")


class NewsTests(unittest.TestCase):
    def test_minimal(self):
        n = News(
            as_of=_now(),
            source=_src(),
            headline="Fed holds rates",
            event_type="monetary_policy",
            timestamp=_now(),
        )
        self.assertEqual(n.item_type, "news")
        self.assertEqual(n.sentiment, "neutral")
        self.assertEqual(n.materiality, "medium")
        self.assertEqual(n.entities, [])

    def test_sentiment_enum(self):
        with self.assertRaises(ValidationError):
            News(
                as_of=_now(),
                source=_src(),
                headline="x",
                event_type="x",
                timestamp=_now(),
                sentiment="ecstatic",  # type: ignore[arg-type]
            )

    def test_retrieval_projection_is_not_a_domain_method(self):
        n = News(
            as_of=_now(),
            source=_src(),
            headline="Fed holds rates",
            event_type="monetary_policy",
            timestamp=_now(),
            entities=["FOMC", "USD"],
        )
        self.assertFalse(hasattr(n, "embedding_text"))


if __name__ == "__main__":
    unittest.main()
