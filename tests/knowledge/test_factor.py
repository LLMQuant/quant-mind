"""Tests for knowledge.factor (stub schema)."""

import unittest
from datetime import datetime, timezone

from quantmind.knowledge._base import SourceRef
from quantmind.knowledge.factor import Factor


def _now() -> datetime:
    return datetime(2026, 4, 27, tzinfo=timezone.utc)


def _src() -> SourceRef:
    return SourceRef(kind="manual")


class FactorTests(unittest.TestCase):
    def test_minimal(self):
        f = Factor(as_of=_now(), source=_src(), factor_name="momentum_12_1")
        self.assertEqual(f.item_type, "factor")
        self.assertEqual(f.factor_name, "momentum_12_1")
        self.assertIsNone(f.universe)

    def test_embedding_text(self):
        f = Factor(
            as_of=_now(),
            source=_src(),
            factor_name="value",
            universe="us_equities",
        )
        self.assertEqual(f.embedding_text(), "factor value on us_equities")

    def test_embedding_text_default_universe(self):
        f = Factor(as_of=_now(), source=_src(), factor_name="size")
        self.assertEqual(f.embedding_text(), "factor size on unspecified")


if __name__ == "__main__":
    unittest.main()
