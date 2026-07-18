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

    def test_retrieval_projection_is_not_a_domain_method(self):
        f = Factor(
            as_of=_now(),
            source=_src(),
            factor_name="value",
            universe="us_equities",
        )
        self.assertFalse(hasattr(f, "embedding_text"))


if __name__ == "__main__":
    unittest.main()
