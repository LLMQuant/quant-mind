"""Tests for ``quantmind.flows._paper_draft``."""

import unittest
from datetime import date

from pydantic import ValidationError

from quantmind.flows._paper_draft import DraftCitation, DraftNode, DraftPaper


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
