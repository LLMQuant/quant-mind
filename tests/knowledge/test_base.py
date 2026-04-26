"""Tests for knowledge._base — BaseKnowledge data standard."""

import unittest
from datetime import datetime, timezone

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


class PackageExportTests(unittest.TestCase):
    def test_top_level_imports(self):
        from quantmind.knowledge import (
            BaseKnowledge,
            Citation,
            Earnings,
            ExtractionRef,
            FlattenKnowledge,
            GraphKnowledge,
            News,
            Paper,
            PaperKnowledgeCard,
            SourceRef,
            TreeKnowledge,
            TreeNode,
        )

        self.assertTrue(issubclass(FlattenKnowledge, BaseKnowledge))
        self.assertTrue(issubclass(TreeKnowledge, BaseKnowledge))
        self.assertTrue(issubclass(GraphKnowledge, BaseKnowledge))
        self.assertTrue(issubclass(News, FlattenKnowledge))
        self.assertTrue(issubclass(Earnings, FlattenKnowledge))
        self.assertTrue(issubclass(PaperKnowledgeCard, FlattenKnowledge))
        self.assertTrue(issubclass(Paper, TreeKnowledge))
        # Ensure side-imports are real classes
        self.assertEqual(Citation.__name__, "Citation")
        self.assertEqual(SourceRef.__name__, "SourceRef")
        self.assertEqual(ExtractionRef.__name__, "ExtractionRef")
        self.assertEqual(TreeNode.__name__, "TreeNode")


if __name__ == "__main__":
    unittest.main()
