"""Tests for slug -> UUID canonicalisation at the LLM extraction boundary.

The openai-agents SDK, with ``strict_json_schema=False``, lets the model emit
human-readable slug ids ("root", "intro", "methodology") wherever the
``Paper`` / ``TreeKnowledge`` schema declares a ``UUID``. The domain model
keeps ``UUID`` (load-bearing: ``tests/knowledge`` assert node-id uniqueness and
UUID-typed JSON round-trips, and the standard relies on stable unique identity
for dedup across re-runs). The
extraction model ``PaperExtraction`` bridges the two: it accepts the slug tree
the LLM produces and canonicalises every id slot to a real ``UUID`` before the
frozen domain validation runs, preserving the tree's structure.
"""

import unittest
from uuid import UUID, uuid4

from quantmind.knowledge import Paper
from quantmind.knowledge.paper import PaperExtraction


def _slug_raw() -> dict:
    """A slug-keyed Paper payload mirroring real gpt-5.4-mini output."""
    return {
        "id": "paper-2606.05138",
        "item_type": "paper",
        "as_of": "2026-05-01T00:00:00Z",
        "source": {"kind": "arxiv", "uri": "arxiv:2606.05138"},
        "root_node_id": "root",
        "citations": [
            {
                "source_id": "paper:2606.05138",
                "tree_id": "root",
                "node_id": "root",
            }
        ],
        "nodes": {
            "root": {
                "node_id": "root",
                "parent_id": None,
                "title": "Paper",
                "summary": "Top-level summary.",
                "children_ids": ["intro", "methodology"],
                "citations": [
                    {"source_id": "s1", "tree_id": "root", "node_id": "root"}
                ],
            },
            "intro": {
                "node_id": "intro",
                "parent_id": "root",
                "title": "Introduction",
                "summary": "Intro summary.",
                "children_ids": [],
                "citations": [],
            },
            "methodology": {
                "node_id": "methodology",
                "parent_id": "root",
                "title": "Methodology",
                "summary": "Method summary.",
                "children_ids": [],
                "citations": [],
            },
        },
    }


class PaperExtractionCanonicalisesSlugIds(unittest.TestCase):
    def test_slug_tree_becomes_uuid_paper_with_structure_preserved(
        self,
    ) -> None:
        paper = PaperExtraction.model_validate(_slug_raw())

        # It is a real Paper with UUID identity.
        self.assertIsInstance(paper, Paper)
        self.assertIsInstance(paper.id, UUID)

        # Root resolves and is a genuine UUID present in the node map.
        self.assertIsInstance(paper.root_node_id, UUID)
        self.assertIn(paper.root_node_id, paper.nodes)
        root = paper.nodes[paper.root_node_id]
        self.assertIsNone(root.parent_id)

        # Children are UUIDs, resolve in the map, and point back to root.
        self.assertEqual(len(root.children_ids), 2)
        for child_id in root.children_ids:
            self.assertIsInstance(child_id, UUID)
            self.assertIn(child_id, paper.nodes)
            self.assertEqual(
                paper.nodes[child_id].parent_id, paper.root_node_id
            )

        # A slug used in multiple positions maps to ONE uuid (internal
        # consistency): the "intro" child id equals the intro node's own id.
        intro_node = next(
            n for n in paper.nodes.values() if n.title == "Introduction"
        )
        self.assertIn(intro_node.node_id, root.children_ids)

        # Citation anchors are canonicalised to the same node uuids.
        self.assertEqual(root.citations[0].node_id, paper.root_node_id)
        self.assertEqual(root.citations[0].tree_id, paper.root_node_id)

    def test_existing_uuid_ids_pass_through_unchanged(self) -> None:
        root_id, child_id = uuid4(), uuid4()
        raw = {
            "item_type": "paper",
            "as_of": "2026-05-01T00:00:00Z",
            "source": {"kind": "arxiv", "uri": "arxiv:x"},
            "root_node_id": str(root_id),
            "nodes": {
                str(root_id): {
                    "node_id": str(root_id),
                    "parent_id": None,
                    "title": "Root",
                    "summary": "s",
                    "children_ids": [str(child_id)],
                },
                str(child_id): {
                    "node_id": str(child_id),
                    "parent_id": str(root_id),
                    "title": "Child",
                    "summary": "s",
                    "children_ids": [],
                },
            },
        }
        paper = PaperExtraction.model_validate(raw)
        self.assertEqual(paper.root_node_id, root_id)
        self.assertIn(child_id, paper.nodes)
        self.assertEqual(paper.nodes[child_id].parent_id, root_id)


if __name__ == "__main__":
    unittest.main()
