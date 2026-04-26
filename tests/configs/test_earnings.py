"""Tests for configs.earnings."""

import unittest

from pydantic import TypeAdapter, ValidationError

from quantmind.configs.earnings import (
    EarningsFlowCfg,
    EarningsInput,
    HttpUrl,
    TickerPeriod,
    TranscriptText,
)


class EarningsFlowCfgTests(unittest.TestCase):
    def test_defaults(self):
        cfg = EarningsFlowCfg()
        self.assertEqual(cfg.model, "gpt-4o")
        self.assertTrue(cfg.detect_surprises)


class EarningsInputTests(unittest.TestCase):
    def setUp(self):
        self.adapter = TypeAdapter(EarningsInput)

    def test_ticker_period(self):
        v = self.adapter.validate_python(
            {
                "type": "ticker_period",
                "ticker": "AAPL",
                "period": "2026Q1",
            }
        )
        self.assertIsInstance(v, TickerPeriod)
        self.assertEqual(v.ticker, "AAPL")

    def test_transcript(self):
        v = self.adapter.validate_python(
            {"type": "transcript", "text": "Operator: ..."}
        )
        self.assertIsInstance(v, TranscriptText)

    def test_http(self):
        v = self.adapter.validate_python(
            {"type": "http", "url": "https://ir.example.com/q1.pdf"}
        )
        self.assertIsInstance(v, HttpUrl)

    def test_unknown_rejected(self):
        with self.assertRaises(ValidationError):
            self.adapter.validate_python({"type": "video", "url": "x"})


if __name__ == "__main__":
    unittest.main()
