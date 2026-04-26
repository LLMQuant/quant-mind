"""Tests for knowledge.paper — Paper (Tree) + PaperKnowledgeCard (Flatten)."""

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from quantmind.knowledge._base import SourceRef
from quantmind.knowledge._tree import TreeNode
from quantmind.knowledge.paper import Paper, PaperKnowledgeCard


def _now() -> datetime:
    return datetime(2026, 4, 1, tzinfo=timezone.utc)


def _src() -> SourceRef:
    return SourceRef(kind="arxiv", uri="arxiv:2604.12345")


def _single_node_paper(**overrides) -> Paper:
    root_id = uuid4()
    root = TreeNode(
        node_id=root_id,
        parent_id=None,
        position=0,
        title="On Cross-Sectional Momentum",
        summary="Methodology for momentum factor on US equities.",
    )
    return Paper(
        as_of=_now(),
        source=_src(),
        root_node_id=root_id,
        nodes={root_id: root},
        **overrides,
    )


class PaperTreeTests(unittest.TestCase):
    def test_minimal(self):
        p = _single_node_paper()
        self.assertEqual(p.item_type, "paper")
        self.assertIsNone(p.arxiv_id)
        self.assertEqual(p.authors, [])
        self.assertEqual(p.asset_classes, [])
        self.assertEqual(p.root().title, "On Cross-Sectional Momentum")

    def test_metadata(self):
        p = _single_node_paper(
            arxiv_id="2604.12345",
            authors=["A. Smith", "B. Jones"],
            asset_classes=["equities"],
        )
        self.assertEqual(p.arxiv_id, "2604.12345")
        self.assertEqual(p.authors, ["A. Smith", "B. Jones"])
        self.assertEqual(p.asset_classes, ["equities"])

    def test_embedding_text_uses_root(self):
        p = _single_node_paper()
        self.assertEqual(
            p.embedding_text(),
            "On Cross-Sectional Momentum\n"
            "Methodology for momentum factor on US equities.",
        )


class PaperKnowledgeCardTests(unittest.TestCase):
    def test_minimal(self):
        paper_id = uuid4()
        card = PaperKnowledgeCard(
            as_of=_now(),
            source=_src(),
            paper_id=paper_id,
            summary="A momentum study on US equities.",
        )
        self.assertEqual(card.item_type, "paper_card")
        self.assertEqual(card.paper_id, paper_id)
        self.assertEqual(card.key_findings, [])
        self.assertEqual(card.asset_classes, [])

    def test_full(self):
        card = PaperKnowledgeCard(
            as_of=_now(),
            source=_src(),
            paper_id=uuid4(),
            summary="s",
            methodology="m",
            key_findings=["f1", "f2"],
            limitations=["l1"],
            asset_classes=["equities"],
        )
        self.assertEqual(card.methodology, "m")
        self.assertEqual(card.key_findings, ["f1", "f2"])

    def test_embedding_text(self):
        card = PaperKnowledgeCard(
            as_of=_now(),
            source=_src(),
            paper_id=uuid4(),
            summary="momentum study",
            key_findings=["beats SPX", "robust to costs"],
        )
        self.assertEqual(
            card.embedding_text(),
            "momentum study\nbeats SPX robust to costs",
        )


if __name__ == "__main__":
    unittest.main()
