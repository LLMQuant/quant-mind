"""Tests for the shared structure tree and paper artifact binding."""

import hashlib
import unittest
from uuid import uuid4

from quantmind.knowledge import (
    Citation,
    PaperStructureNodeDraft,
    PaperStructureProducer,
    PaperStructureTree,
    PaperStructureTreeDraft,
    StructureTree,
    StructureTreeValidationError,
)
from tests.paper_helpers import build_paper_result, build_paper_structure_tree


class PaperStructureTreeTests(unittest.TestCase):
    def test_from_draft_builds_valid_cited_tree_without_content(self) -> None:
        tree = build_paper_structure_tree()

        tree.validate(source_pages={1, 2})
        self.assertEqual(len(tree.nodes), 3)
        self.assertEqual(
            [node.title for node in tree.walk_dfs()],
            [
                "Attention Is All You Need",
                "Architecture",
                "Attention and results",
            ],
        )
        self.assertTrue(
            all(
                node.content is None and node.citations
                for node in tree.nodes.values()
            )
        )
        self.assertEqual(
            {citation.page for citation in tree.root().citations},
            {1, 2},
        )
        self.assertTrue(
            all(
                citation.node_id is None and citation.tree_id is None
                for node in tree.nodes.values()
                for citation in node.citations
            )
        )

    def test_identity_is_idempotent_and_producer_changes_version_it(
        self,
    ) -> None:
        first = build_paper_structure_tree()
        repeated = build_paper_structure_tree()
        changed = build_paper_structure_tree(model="another-model")

        self.assertEqual(first.id, repeated.id)
        self.assertEqual(first.root_node_id, repeated.root_node_id)
        self.assertEqual(first.nodes, repeated.nodes)
        self.assertNotEqual(first.id, changed.id)

    def test_low_quality_draft_falls_back_to_flat_source_order(self) -> None:
        tree = build_paper_structure_tree(quality="low")

        self.assertEqual(len(tree.children_of(tree.root_node_id)), 2)
        self.assertEqual(
            [node.position for node in tree.children_of(tree.root_node_id)],
            [0, 1],
        )

    def test_page_outside_exact_source_is_rejected(self) -> None:
        result = build_paper_result()
        producer = PaperStructureProducer(
            model="fake",
            prompt_version="test-v1",
            instructions_hash=hashlib.sha256(b"instructions").hexdigest(),
            page_text_chars=1_200,
            max_output_tokens=256,
            max_depth=2,
            max_nodes=4,
        )
        draft = PaperStructureTreeDraft(
            root=PaperStructureNodeDraft(
                title="Paper",
                summary="Paper summary.",
                start_page=1,
                end_page=99,
            )
        )

        with self.assertRaisesRegex(
            StructureTreeValidationError,
            "outside its source",
        ):
            PaperStructureTree.from_draft(
                result.source_revision,
                producer=producer,
                draft=draft,
            )


class StructureTreeIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        artifact = build_paper_structure_tree()
        self.structure = StructureTree(
            root_node_id=artifact.root_node_id,
            nodes=artifact.nodes,
        )
        self.root = self.structure.root()
        self.first, self.second = self.structure.children_of(
            self.structure.root_node_id
        )

    def _with_nodes(self, *nodes) -> StructureTree:
        values = dict(self.structure.nodes)
        values.update({node.node_id: node for node in nodes})
        return StructureTree(
            root_node_id=self.structure.root_node_id,
            nodes=values,
        )

    def test_rejects_multiple_roots(self) -> None:
        invalid = self._with_nodes(
            self.first.model_copy(update={"parent_id": None})
        )
        with self.assertRaisesRegex(
            StructureTreeValidationError,
            "exactly one",
        ):
            invalid.validate(source_pages={1, 2})

    def test_rejects_cycle(self) -> None:
        root = self.root.model_copy(update={"children_ids": []})
        first = self.first.model_copy(
            update={
                "parent_id": self.second.node_id,
                "children_ids": [self.second.node_id],
            }
        )
        second = self.second.model_copy(
            update={
                "parent_id": self.first.node_id,
                "children_ids": [self.first.node_id],
            }
        )
        invalid = self._with_nodes(root, first, second)
        with self.assertRaisesRegex(StructureTreeValidationError, "cycle"):
            invalid.validate(source_pages={1, 2})

    def test_rejects_orphan(self) -> None:
        orphan = self.first.model_copy(update={"parent_id": uuid4()})
        invalid = self._with_nodes(orphan)
        with self.assertRaisesRegex(StructureTreeValidationError, "orphan"):
            invalid.validate(source_pages={1, 2})

    def test_rejects_inconsistent_parent_child_links(self) -> None:
        root = self.root.model_copy(
            update={"children_ids": [self.second.node_id]}
        )
        invalid = self._with_nodes(root)
        with self.assertRaisesRegex(
            StructureTreeValidationError,
            "inconsistent",
        ):
            invalid.validate(source_pages={1, 2})

    def test_rejects_duplicate_sibling_positions(self) -> None:
        second = self.second.model_copy(
            update={"position": self.first.position}
        )
        invalid = self._with_nodes(second)
        with self.assertRaisesRegex(
            StructureTreeValidationError,
            "duplicate positions",
        ):
            invalid.validate(source_pages={1, 2})

    def test_rejects_page_outside_source(self) -> None:
        citation = self.first.citations[0]
        outside = Citation(
            source_id=citation.source_id,
            page=99,
            tree_id=citation.tree_id,
            node_id=citation.node_id,
        )
        first = self.first.model_copy(update={"citations": [outside]})
        invalid = self._with_nodes(first)
        with self.assertRaisesRegex(
            StructureTreeValidationError,
            "outside its source",
        ):
            invalid.validate(source_pages={1, 2})

    def test_rejects_child_pages_not_contained_by_parent(self) -> None:
        root_citations = [
            citation for citation in self.root.citations if citation.page == 1
        ]
        root = self.root.model_copy(update={"citations": root_citations})
        invalid = self._with_nodes(root)
        with self.assertRaisesRegex(
            StructureTreeValidationError,
            "not contained",
        ):
            invalid.validate(source_pages={1, 2})


if __name__ == "__main__":
    unittest.main()
