"""Tests for knowledge._base — BaseKnowledge data standard."""

import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pydantic import ValidationError

from quantmind.knowledge._base import (
    BaseKnowledge,
    Citation,
    ExtractionRef,
    SourceRef,
)


def _now() -> datetime:
    return datetime(2026, 4, 26, tzinfo=timezone.utc)


def _src() -> SourceRef:
    return SourceRef(kind="manual")


class CitationTests(unittest.TestCase):
    def test_minimal(self):
        cit = Citation(source_id="arxiv:2604.12345")
        self.assertEqual(cit.source_id, "arxiv:2604.12345")
        self.assertIsNone(cit.page)
        self.assertIsNone(cit.quote)
        self.assertIsNone(cit.tree_id)
        self.assertIsNone(cit.node_id)

    def test_quote_max_length(self):
        with self.assertRaises(ValidationError):
            Citation(source_id="x", quote="a" * 501)

    def test_tree_anchor_round_trip(self):
        tree_id = uuid4()
        node_id = uuid4()
        cit = Citation(source_id="paper:abc", tree_id=tree_id, node_id=node_id)
        self.assertEqual(cit.tree_id, tree_id)
        self.assertEqual(cit.node_id, node_id)
        # JSON round-trip preserves UUID anchors.
        revived = Citation.model_validate_json(cit.model_dump_json())
        self.assertEqual(revived.tree_id, tree_id)
        self.assertEqual(revived.node_id, node_id)


class SourceRefTests(unittest.TestCase):
    def test_minimal(self):
        s = SourceRef(kind="arxiv", uri="arxiv:2604.12345")
        self.assertEqual(s.kind, "arxiv")
        self.assertEqual(s.uri, "arxiv:2604.12345")
        self.assertIsNone(s.fetched_at)
        self.assertIsNone(s.content_hash)

    def test_kind_enum_enforced(self):
        with self.assertRaises(ValidationError):
            SourceRef(kind="ftp")  # type: ignore[arg-type]

    def test_extra_forbidden(self):
        with self.assertRaises(ValidationError):
            SourceRef(kind="manual", garbage=1)  # type: ignore[call-arg]


class ExtractionRefTests(unittest.TestCase):
    def test_minimal(self):
        e = ExtractionRef(
            flow="paper_flow", model="gpt-4o", extracted_at=_now()
        )
        self.assertEqual(e.flow, "paper_flow")
        self.assertEqual(e.model, "gpt-4o")
        self.assertIsNone(e.run_id)


class _ConcreteKnowledge(BaseKnowledge):
    """Test fixture: concrete subclass that overrides embedding_text."""

    item_type: str = "test"  # pyright: ignore[reportIncompatibleVariableOverride]
    payload: str = ""

    def embedding_text(self) -> str:
        return self.payload


class BaseKnowledgeTests(unittest.TestCase):
    def test_as_of_required(self):
        with self.assertRaises(ValidationError):
            _ConcreteKnowledge(source=_src())  # type: ignore[call-arg]

    def test_source_required(self):
        with self.assertRaises(ValidationError):
            _ConcreteKnowledge(as_of=_now())  # type: ignore[call-arg]

    def test_default_id_unique(self):
        a = _ConcreteKnowledge(as_of=_now(), source=_src())
        b = _ConcreteKnowledge(as_of=_now(), source=_src())
        self.assertNotEqual(a.id, b.id)

    def test_default_confidence_is_medium(self):
        item = _ConcreteKnowledge(as_of=_now(), source=_src())
        self.assertEqual(item.confidence, "medium")

    def test_default_schema_version(self):
        item = _ConcreteKnowledge(as_of=_now(), source=_src())
        self.assertEqual(item.schema_version, "1.0")

    def test_created_at_auto(self):
        before = datetime.now(timezone.utc)
        item = _ConcreteKnowledge(as_of=_now(), source=_src())
        after = datetime.now(timezone.utc)
        self.assertGreaterEqual(item.created_at, before)
        self.assertLessEqual(item.created_at, after)

    def test_frozen(self):
        item = _ConcreteKnowledge(as_of=_now(), source=_src())
        with self.assertRaises(ValidationError):
            item.tags = ["new"]  # type: ignore[misc]

    def test_extra_forbidden(self):
        with self.assertRaises(ValidationError):
            _ConcreteKnowledge(
                as_of=_now(),
                source=_src(),
                unexpected_field=1,  # type: ignore[call-arg]
            )

    def test_embedding_text_default_raises(self):
        # BaseKnowledge.embedding_text raises NotImplementedError; subclasses
        # must override. We test via a class that doesn't override.
        class _NoEmbed(BaseKnowledge):
            item_type: str = "no_embed"  # pyright: ignore[reportIncompatibleVariableOverride]

        item = _NoEmbed(as_of=_now(), source=_src())
        with self.assertRaises(NotImplementedError):
            item.embedding_text()

    def test_embedding_text_override(self):
        item = _ConcreteKnowledge(as_of=_now(), source=_src(), payload="hello")
        self.assertEqual(item.embedding_text(), "hello")

    def test_extraction_optional(self):
        item = _ConcreteKnowledge(as_of=_now(), source=_src())
        self.assertIsNone(item.extraction)

    def test_extraction_round_trip(self):
        ext = ExtractionRef(
            flow="paper_flow", model="gpt-4o", extracted_at=_now()
        )
        item = _ConcreteKnowledge(as_of=_now(), source=_src(), extraction=ext)
        assert item.extraction is not None
        self.assertEqual(item.extraction.flow, "paper_flow")

    def test_is_extracted_false_when_hand_curated(self):
        item = _ConcreteKnowledge(as_of=_now(), source=_src())
        self.assertFalse(item.is_extracted())

    def test_is_extracted_true_when_extraction_set(self):
        ext = ExtractionRef(
            flow="paper_flow", model="gpt-4o", extracted_at=_now()
        )
        item = _ConcreteKnowledge(as_of=_now(), source=_src(), extraction=ext)
        self.assertTrue(item.is_extracted())

    def test_freshness_with_explicit_now(self):
        item = _ConcreteKnowledge(as_of=_now(), source=_src())
        future = _now() + timedelta(days=3)
        self.assertEqual(item.freshness(future), timedelta(days=3))

    def test_freshness_default_now_is_utc(self):
        # Just verify it returns a timedelta without raising; default `now`
        # comes from `datetime.now(timezone.utc)`.
        item = _ConcreteKnowledge(as_of=_now(), source=_src())
        self.assertIsInstance(item.freshness(), timedelta)

    def test_with_tags_appends_unique(self):
        item = _ConcreteKnowledge(as_of=_now(), source=_src(), tags=["macro"])
        updated = item.with_tags("equities", "macro", "rates")
        self.assertEqual(updated.tags, ["macro", "equities", "rates"])
        # Original is frozen and unchanged.
        self.assertEqual(item.tags, ["macro"])

    def test_with_tags_returns_new_instance(self):
        item = _ConcreteKnowledge(as_of=_now(), source=_src())
        updated = item.with_tags("x")
        self.assertIsNot(item, updated)
        self.assertEqual(updated.tags, ["x"])


class PackageExportTests(unittest.TestCase):
    def test_top_level_imports(self):
        from quantmind.knowledge import (
            BaseKnowledge,
            Citation,
            Earnings,
            ExtractionRef,
            Factor,
            FlattenKnowledge,
            GraphKnowledge,
            News,
            Paper,
            PaperKnowledgeCard,
            SourceRef,
            Thesis,
            TreeKnowledge,
            TreeNode,
        )

        self.assertTrue(issubclass(FlattenKnowledge, BaseKnowledge))
        self.assertTrue(issubclass(TreeKnowledge, BaseKnowledge))
        self.assertTrue(issubclass(GraphKnowledge, BaseKnowledge))
        self.assertTrue(issubclass(News, FlattenKnowledge))
        self.assertTrue(issubclass(Earnings, FlattenKnowledge))
        self.assertTrue(issubclass(PaperKnowledgeCard, FlattenKnowledge))
        self.assertTrue(issubclass(Factor, FlattenKnowledge))
        self.assertTrue(issubclass(Thesis, FlattenKnowledge))
        self.assertTrue(issubclass(Paper, TreeKnowledge))
        # Ensure side-imports are real classes
        self.assertEqual(Citation.__name__, "Citation")
        self.assertEqual(SourceRef.__name__, "SourceRef")
        self.assertEqual(ExtractionRef.__name__, "ExtractionRef")
        self.assertEqual(TreeNode.__name__, "TreeNode")


if __name__ == "__main__":
    unittest.main()
