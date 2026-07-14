"""Tests for the intent-oriented news configuration."""

import unittest
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from quantmind.configs import NewsFlowCfg, NewsWindow


class NewsFlowCfgTests(unittest.TestCase):
    def test_raw_html_is_not_retained_by_default(self) -> None:
        self.assertFalse(NewsFlowCfg().retain_raw_html)


class NewsWindowTests(unittest.TestCase):
    def test_normalizes_aware_timestamps_to_utc(self) -> None:
        window = NewsWindow(
            source="pr-newswire",
            start=datetime(
                2026,
                7,
                13,
                8,
                tzinfo=timezone(timedelta(hours=8)),
            ),
            end=datetime(
                2026,
                7,
                14,
                8,
                tzinfo=timezone(timedelta(hours=8)),
            ),
        )

        self.assertEqual(window.start.tzinfo, timezone.utc)
        self.assertEqual(window.start.hour, 0)
        self.assertEqual(window.end.tzinfo, timezone.utc)

    def test_rejects_naive_timestamps(self) -> None:
        with self.assertRaises(ValidationError):
            NewsWindow(
                source="pr-newswire",
                start=datetime(2026, 7, 13),
                end=datetime(2026, 7, 14, tzinfo=timezone.utc),
            )

    def test_rejects_empty_or_reversed_window(self) -> None:
        timestamp = datetime(2026, 7, 14, tzinfo=timezone.utc)

        with self.assertRaisesRegex(ValidationError, "end to be after start"):
            NewsWindow(
                source="pr-newswire",
                start=timestamp,
                end=timestamp,
            )

    def test_rejects_unsupported_source(self) -> None:
        with self.assertRaises(ValidationError):
            NewsWindow.model_validate(
                {
                    "source": "business-wire",
                    "start": datetime(2026, 7, 13, tzinfo=timezone.utc),
                    "end": datetime(2026, 7, 14, tzinfo=timezone.utc),
                }
            )


if __name__ == "__main__":
    unittest.main()
