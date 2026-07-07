"""Tests for ``quantmind.flows._paper_draft``."""

import unittest
from datetime import date, datetime, timezone
from uuid import UUID

from pydantic import ValidationError

from quantmind.flows._paper_draft import (
    DraftCitation,
    DraftNode,
    DraftPaper,
    assemble_paper,
)
from quantmind.knowledge import ExtractionRef, Paper, SourceRef


class DraftSchemaTests(unittest.TestCase):
    def test_parses_nested_tree(self) -> None:
        draft = DraftPaper.model_validate(
            {
                "title": "Momentum",
                "summary": "top",
                "published_date": "2023-12-07",
                "authors": ["Alice"],
                "root": {
                    "title": "Root",
                    "summary": "root summary",
                    "children": [
                        {"title": "Intro", "summary": "s", "content": "body"}
                    ],
                },
            }
        )
        self.assertEqual(draft.published_date, date(2023, 12, 7))
        self.assertEqual(draft.root.children[0].title, "Intro")
        self.assertEqual(draft.root.children[0].content, "body")

    def test_extra_fields_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            DraftNode.model_validate(
                {"title": "x", "summary": "y", "node_id": "abc"}
            )

    def test_citation_has_no_id_fields(self) -> None:
        cite = DraftCitation(quote="q", page=3)
        self.assertEqual(cite.page, 3)
        self.assertNotIn("node_id", DraftCitation.model_fields)
        self.assertNotIn("tree_id", DraftCitation.model_fields)


def _args() -> dict:
    return {
        "source": SourceRef(kind="local", uri="/tmp/p.pdf"),
        "source_id": "p.pdf",
        "as_of": datetime(2023, 12, 7, tzinfo=timezone.utc),
        "extraction": ExtractionRef(
            flow="paper_flow",
            model="gpt-4o-mini",
            extracted_at=datetime(2023, 12, 7, tzinfo=timezone.utc),
        ),
        "out_type": Paper,
    }


class AssemblePaperTests(unittest.TestCase):
    def test_single_root(self) -> None:
        draft = DraftPaper(
            title="T",
            summary="s",
            root=DraftNode(title="Root", summary="rs", content="body"),
        )
        paper = assemble_paper(draft, **_args())
        self.assertIsInstance(paper, Paper)
        self.assertEqual(len(paper.nodes), 1)
        self.assertEqual(paper.root_node_id, paper.root().node_id)
        self.assertEqual(paper.root().title, "Root")
        self.assertEqual(paper.source.kind, "local")
        self.assertEqual(
            paper.as_of, datetime(2023, 12, 7, tzinfo=timezone.utc)
        )
        self.assertIsInstance(paper.root().node_id, UUID)

    def test_nested_wiring_and_positions(self) -> None:
        draft = DraftPaper(
            title="T",
            summary="s",
            root=DraftNode(
                title="Root",
                summary="rs",
                children=[
                    DraftNode(
                        title="A",
                        summary="a",
                        children=[DraftNode(title="A1", summary="a1")],
                    ),
                    DraftNode(title="B", summary="b"),
                ],
            ),
        )
        paper = assemble_paper(draft, **_args())
        self.assertEqual(len(paper.nodes), 4)
        titles = [n.title for n in paper.walk_dfs()]
        self.assertEqual(titles, ["Root", "A", "A1", "B"])
        root = paper.root()
        self.assertEqual(len(root.children_ids), 2)
        a = paper.children_of(root.node_id)[0]
        self.assertEqual(a.parent_id, root.node_id)
        self.assertEqual(a.position, 0)
        b = paper.children_of(root.node_id)[1]
        self.assertEqual(b.position, 1)
        # uuids unique across the tree
        ids = [n.node_id for n in paper.walk_dfs()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_citations_wired_to_node(self) -> None:
        draft = DraftPaper(
            title="T",
            summary="s",
            root=DraftNode(
                title="Root",
                summary="rs",
                citations=[DraftCitation(quote="q", page=5)],
            ),
        )
        paper = assemble_paper(draft, **_args())
        cite = paper.root().citations[0]
        self.assertEqual(cite.quote, "q")
        self.assertEqual(cite.page, 5)
        self.assertEqual(cite.source_id, "p.pdf")
        self.assertEqual(cite.node_id, paper.root().node_id)
        self.assertEqual(cite.tree_id, paper.id)

    def test_out_type_subclass(self) -> None:
        class MyPaper(Paper):
            pass

        draft = DraftPaper(
            title="T",
            summary="s",
            root=DraftNode(title="Root", summary="rs"),
        )
        args = _args()
        args["out_type"] = MyPaper
        paper = assemble_paper(draft, **args)
        self.assertIsInstance(paper, MyPaper)
