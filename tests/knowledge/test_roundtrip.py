"""JSON round-trip tests for every concrete knowledge schema.

These tests guard the contract that powers ``Agent(output_type=...)``: the
LLM returns JSON, the Agents SDK calls ``model_validate_json`` on it, and
the result must equal what we would have produced via ``model_dump_json``.
Tree schemas are the trickiest because of ``dict[UUID, TreeNode]`` keys.
"""

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from quantmind.knowledge import (
    Earnings,
    ExtractionRef,
    Factor,
    News,
    Paper,
    PaperKnowledgeCard,
    SourceRef,
    Thesis,
    TreeNode,
)


def _now() -> datetime:
    return datetime(2026, 4, 27, tzinfo=timezone.utc)


def _src(kind: str = "manual") -> SourceRef:
    return SourceRef(kind=kind)  # type: ignore[arg-type]


def _ext() -> ExtractionRef:
    return ExtractionRef(flow="test_flow", model="gpt-4o", extracted_at=_now())


class FlattenRoundTripTests(unittest.TestCase):
    def test_news(self):
        n = News(
            as_of=_now(),
            source=_src("rss"),
            extraction=_ext(),
            headline="Fed holds rates",
            event_type="monetary_policy",
            timestamp=_now(),
            entities=["FOMC"],
        )
        revived = News.model_validate_json(n.model_dump_json())
        self.assertEqual(n, revived)

    def test_earnings(self):
        e = Earnings(
            as_of=_now(),
            source=_src("http"),
            ticker="AAPL",
            period="2026Q1",
            revenue=120.0,
            eps=1.55,
            surprise_flags=["eps_beat"],
        )
        revived = Earnings.model_validate_json(e.model_dump_json())
        self.assertEqual(e, revived)

    def test_paper_knowledge_card(self):
        card = PaperKnowledgeCard(
            as_of=_now(),
            source=_src("arxiv"),
            paper_id=uuid4(),
            summary="momentum study",
            methodology="cross-sectional",
            key_findings=["beats SPX"],
            asset_classes=["equities"],
        )
        revived = PaperKnowledgeCard.model_validate_json(card.model_dump_json())
        self.assertEqual(card, revived)

    def test_factor(self):
        f = Factor(
            as_of=_now(),
            source=_src(),
            factor_name="value",
            universe="us_equities",
        )
        revived = Factor.model_validate_json(f.model_dump_json())
        self.assertEqual(f, revived)

    def test_thesis(self):
        t = Thesis(as_of=_now(), source=_src(), claim="USD softens")
        revived = Thesis.model_validate_json(t.model_dump_json())
        self.assertEqual(t, revived)


class TreeRoundTripTests(unittest.TestCase):
    def _build_paper(self) -> Paper:
        leaf_id = uuid4()
        root_id = uuid4()
        leaf = TreeNode(
            node_id=leaf_id,
            parent_id=root_id,
            position=0,
            title="Methodology",
            summary="cross-sectional momentum signal",
            content="We rank stocks by 12-1 month returns ...",
        )
        root = TreeNode(
            node_id=root_id,
            parent_id=None,
            position=0,
            title="Cross-Sectional Momentum",
            summary="paper-level summary",
            children_ids=[leaf_id],
        )
        return Paper(
            as_of=_now(),
            source=_src("arxiv"),
            extraction=_ext(),
            arxiv_id="2604.12345",
            authors=["A. Smith"],
            asset_classes=["equities"],
            root_node_id=root_id,
            nodes={root_id: root, leaf_id: leaf},
        )

    def test_paper_dict_uuid_keys_round_trip(self):
        p = self._build_paper()
        revived = Paper.model_validate_json(p.model_dump_json())
        self.assertEqual(p, revived)
        # UUID keys are preserved through the JSON detour.
        self.assertEqual(set(revived.nodes.keys()), set(p.nodes.keys()))
        # Helper still works on the revived instance.
        self.assertEqual(revived.root().title, "Cross-Sectional Momentum")

    def test_paper_walk_dfs_after_round_trip(self):
        p = self._build_paper()
        revived = Paper.model_validate_json(p.model_dump_json())
        titles = [n.title for n in revived.walk_dfs()]
        self.assertEqual(titles, ["Cross-Sectional Momentum", "Methodology"])


if __name__ == "__main__":
    unittest.main()
