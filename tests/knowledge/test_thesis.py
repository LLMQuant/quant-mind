"""Tests for knowledge.thesis (stub schema)."""

import unittest
from datetime import datetime, timezone

from quantmind.knowledge._base import SourceRef
from quantmind.knowledge.thesis import Thesis


def _now() -> datetime:
    return datetime(2026, 4, 27, tzinfo=timezone.utc)


def _src() -> SourceRef:
    return SourceRef(kind="manual")


class ThesisTests(unittest.TestCase):
    def test_minimal(self):
        t = Thesis(as_of=_now(), source=_src(), claim="USD weakens in H2 2026")
        self.assertEqual(t.item_type, "thesis")
        self.assertEqual(t.claim, "USD weakens in H2 2026")

    def test_embedding_text_returns_claim(self):
        t = Thesis(as_of=_now(), source=_src(), claim="rates higher for longer")
        self.assertEqual(t.embedding_text(), "rates higher for longer")


if __name__ == "__main__":
    unittest.main()
