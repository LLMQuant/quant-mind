"""Tests for ``quantmind.flows._paper_draft``.

The draft layer is the LLM-facing extraction schema (strict-structured-output
safe: nested children, plain-string ids) plus ``draft_to_paper`` which lifts a
validated draft into the canonical ``quantmind.knowledge.Paper`` store schema
(UUID identity, flat ``nodes`` map, injected provenance).
"""

import unittest
from datetime import datetime, timezone

from quantmind.flows._paper_draft import (
    DraftCitation,
    PaperDraft,
    PaperDraftNode,
    draft_to_paper,
)
from quantmind.knowledge import Paper

_ARXIV_META = {
    "source": "arxiv",
    "arxiv_id": "2604.12345",
    "title": "Cross-Sectional Momentum",
    "authors": ["Alice", "Bob"],
}


def _leaf(
    title: str, summary: str = "s", content: str | None = "body"
) -> PaperDraftNode:
    return PaperDraftNode(title=title, summary=summary, content=content)


class DraftToPaperStructureTests(unittest.TestCase):
    def test_single_node_draft_becomes_valid_paper(self) -> None:
        draft = PaperDraft(root=PaperDraftNode(title="Root", summary="top"))
        paper = draft_to_paper(
            draft, source_meta=_ARXIV_META, model="gpt-4o-mini"
        )
        self.assertIsInstance(paper, Paper)
        # Root resolves and carries the draft's title/summary.
        self.assertEqual(paper.root().title, "Root")
        self.assertEqual(paper.root().summary, "top")
        # root_node_id points into the nodes map.
        self.assertIn(paper.root_node_id, paper.nodes)
        self.assertEqual(len(paper.nodes), 1)

    def test_nested_children_get_uuids_and_parent_child_wiring(self) -> None:
        draft = PaperDraft(
            root=PaperDraftNode(
                title="Root",
                summary="top",
                children=[_leaf("Intro"), _leaf("Method")],
            )
        )
        paper = draft_to_paper(
            draft, source_meta=_ARXIV_META, model="gpt-4o-mini"
        )
        root = paper.root()
        self.assertEqual(len(paper.nodes), 3)
        self.assertEqual(len(root.children_ids), 2)
        children = paper.children_of(root.node_id)
        self.assertEqual([c.title for c in children], ["Intro", "Method"])
        # Each child points back to the root and preserves declared order.
        for pos, child in enumerate(children):
            self.assertEqual(child.parent_id, root.node_id)
            self.assertEqual(child.position, pos)
        # Root has no parent.
        self.assertIsNone(root.parent_id)

    def test_leaf_content_preserved(self) -> None:
        draft = PaperDraft(
            root=PaperDraftNode(
                title="Root",
                summary="top",
                children=[_leaf("Body", content="full markdown")],
            )
        )
        paper = draft_to_paper(
            draft, source_meta=_ARXIV_META, model="gpt-4o-mini"
        )
        leaf = paper.children_of(paper.root_node_id)[0]
        self.assertEqual(leaf.content, "full markdown")


class DraftToPaperProvenanceTests(unittest.TestCase):
    def test_source_injected_from_arxiv_meta_not_llm(self) -> None:
        draft = PaperDraft(root=PaperDraftNode(title="R", summary="s"))
        paper = draft_to_paper(
            draft, source_meta=_ARXIV_META, model="gpt-4o-mini"
        )
        self.assertEqual(paper.source.kind, "arxiv")
        self.assertIn("2604.12345", paper.source.uri or "")

    def test_web_meta_maps_to_http_source(self) -> None:
        draft = PaperDraft(root=PaperDraftNode(title="R", summary="s"))
        meta = {
            "source": "web",
            "url": "https://example.com/x.pdf",
            "content_type": "application/pdf",
        }
        paper = draft_to_paper(draft, source_meta=meta, model="m")
        self.assertEqual(paper.source.kind, "http")
        self.assertEqual(paper.source.uri, "https://example.com/x.pdf")

    def test_inline_meta_maps_to_manual_source(self) -> None:
        draft = PaperDraft(root=PaperDraftNode(title="R", summary="s"))
        paper = draft_to_paper(
            draft, source_meta={"source": "inline"}, model="m"
        )
        self.assertEqual(paper.source.kind, "manual")

    def test_extraction_records_flow_and_model(self) -> None:
        draft = PaperDraft(root=PaperDraftNode(title="R", summary="s"))
        paper = draft_to_paper(draft, source_meta=_ARXIV_META, model="gpt-x")
        self.assertIsNotNone(paper.extraction)
        assert paper.extraction is not None  # narrow for type-checker
        self.assertEqual(paper.extraction.flow, "paper_flow")
        self.assertEqual(paper.extraction.model, "gpt-x")

    def test_arxiv_id_and_authors_taken_from_meta(self) -> None:
        # Draft leaves them empty; the flow knows them from fetch metadata.
        draft = PaperDraft(root=PaperDraftNode(title="R", summary="s"))
        paper = draft_to_paper(
            draft, source_meta=_ARXIV_META, model="gpt-4o-mini"
        )
        self.assertEqual(paper.arxiv_id, "2604.12345")
        self.assertEqual(paper.authors, ["Alice", "Bob"])

    def test_asset_classes_taken_from_draft(self) -> None:
        draft = PaperDraft(
            root=PaperDraftNode(title="R", summary="s"),
            asset_classes=["equities", "rates"],
        )
        paper = draft_to_paper(draft, source_meta=_ARXIV_META, model="m")
        self.assertEqual(paper.asset_classes, ["equities", "rates"])

    def test_as_of_uses_published_at_when_present(self) -> None:
        published = datetime(2025, 3, 1, tzinfo=timezone.utc)
        meta = {**_ARXIV_META, "published_at": published}
        draft = PaperDraft(root=PaperDraftNode(title="R", summary="s"))
        paper = draft_to_paper(draft, source_meta=meta, model="m")
        self.assertEqual(paper.as_of, published)

    def test_as_of_defaults_to_aware_now_without_published_at(self) -> None:
        draft = PaperDraft(root=PaperDraftNode(title="R", summary="s"))
        paper = draft_to_paper(draft, source_meta=_ARXIV_META, model="m")
        self.assertIsNotNone(paper.as_of.tzinfo)


class DraftToPaperCitationTests(unittest.TestCase):
    def test_citation_mapped_onto_node(self) -> None:
        draft = PaperDraft(
            root=PaperDraftNode(
                title="R",
                summary="s",
                citations=[
                    DraftCitation(source_id="arxiv:1", page=2, quote="hi")
                ],
            )
        )
        paper = draft_to_paper(draft, source_meta=_ARXIV_META, model="m")
        cites = paper.root().citations
        self.assertEqual(len(cites), 1)
        self.assertEqual(cites[0].source_id, "arxiv:1")
        self.assertEqual(cites[0].page, 2)

    def test_overlong_quote_truncated_to_schema_limit(self) -> None:
        draft = PaperDraft(
            root=PaperDraftNode(
                title="R",
                summary="s",
                citations=[DraftCitation(source_id="x", quote="z" * 600)],
            )
        )
        paper = draft_to_paper(draft, source_meta=_ARXIV_META, model="m")
        self.assertEqual(len(paper.root().citations[0].quote or ""), 500)


if __name__ == "__main__":
    unittest.main()
