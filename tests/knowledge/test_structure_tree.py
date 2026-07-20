"""Tests for the shared structure tree and paper artifact binding."""

import hashlib
import unittest
from datetime import datetime, timezone
from uuid import uuid4

from quantmind.knowledge import (
    Citation,
    PaperStructureNodeDraft,
    PaperStructureProducer,
    PaperStructureTree,
    PaperStructureTreeDraft,
    StructureTree,
    StructureTreeValidationError,
    TreeNode,
)
from quantmind.knowledge.paper import _paper_structure_content_hash
from tests.paper_helpers import build_paper_result, build_paper_structure_tree


class PaperStructureTreeTests(unittest.TestCase):
    def test_from_draft_builds_self_contained_cited_tree(self) -> None:
        tree = build_paper_structure_tree()
        pages = build_paper_result().source_revision.parsed.pages

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
        # Internal nodes carry no content; every leaf carries its own text.
        for node in tree.nodes.values():
            self.assertTrue(node.citations)
            if node.children_ids:
                self.assertIsNone(node.content)
            else:
                self.assertTrue(node.content)
        leaves = {
            node.title: node
            for node in tree.nodes.values()
            if not node.children_ids
        }
        self.assertEqual(leaves["Architecture"].content, pages[0].text)
        self.assertEqual(leaves["Attention and results"].content, pages[1].text)
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

    def test_from_draft_populates_provenance_metadata(self) -> None:
        source = build_paper_result().source_revision
        tree = build_paper_structure_tree()

        # Light provenance is copied from the exact source revision so the tree
        # can be stored and time-queried standalone.
        self.assertEqual(tree.as_of, source.as_of)
        self.assertEqual(tree.source, source.source)
        self.assertEqual(tree.source_title, source.title)
        self.assertEqual(tree.source_content_hash, source.source.content_hash)
        self.assertEqual(
            tree.source.uri, "https://arxiv.org/pdf/1706.03762v7.pdf"
        )

    def test_provenance_is_metadata_not_identity(self) -> None:
        # Same source bytes rebuilt at a different wall-clock time: the source
        # revision differs only in its timestamps, so identity and content hash
        # must be identical while the provenance ``as_of`` differs.
        early = build_paper_structure_tree()
        late = build_paper_structure_tree(
            when=datetime(2020, 1, 1, tzinfo=timezone.utc)
        )

        self.assertEqual(early.id, late.id)
        self.assertEqual(early.content_hash, late.content_hash)
        self.assertEqual(early.root_node_id, late.root_node_id)
        self.assertEqual(early.nodes, late.nodes)
        self.assertNotEqual(early.as_of, late.as_of)

        # Mutating only the provenance also leaves identity and content hash
        # untouched: these fields never enter ``id`` / ``content_hash``.
        shifted = early.model_copy(
            update={"as_of": datetime(1999, 1, 1, tzinfo=timezone.utc)}
        )
        self.assertEqual(shifted.id, early.id)
        self.assertEqual(shifted.content_hash, early.content_hash)

    def test_multi_page_leaf_joins_cited_page_text_in_order(self) -> None:
        result = build_paper_result()
        pages = result.source_revision.parsed.pages
        producer = PaperStructureProducer(
            model="fake",
            prompt_version="test-v1",
            instructions_hash=hashlib.sha256(b"instructions").hexdigest(),
            page_text_chars=1_200,
            max_output_tokens=256,
            max_depth=2,
            max_nodes=4,
        )
        # A single root leaf spanning both physical pages.
        draft = PaperStructureTreeDraft(
            root=PaperStructureNodeDraft(
                title="Paper",
                summary="Whole paper as one leaf.",
                start_page=1,
                end_page=2,
            )
        )
        tree = PaperStructureTree.from_draft(
            result.source_revision, producer=producer, draft=draft
        )
        self.assertEqual(len(tree.nodes), 1)
        self.assertEqual(
            tree.root().content, f"{pages[0].text}\n\n{pages[1].text}"
        )

    def _revalidate_with_nodes(
        self, tree: PaperStructureTree, *updated: TreeNode
    ) -> PaperStructureTree:
        """Rebuild ``tree`` with tampered nodes and re-run its integrity gate."""
        nodes = dict(tree.nodes)
        for node in updated:
            nodes[node.node_id] = node
        payload = tree.model_dump(mode="json")
        payload["content_hash"] = _paper_structure_content_hash(
            tree.root_node_id, nodes
        )
        payload["nodes"] = {
            str(node_id): node.model_dump(mode="json")
            for node_id, node in nodes.items()
        }
        return PaperStructureTree.model_validate(payload)

    def test_rejects_leaf_without_content(self) -> None:
        tree = build_paper_structure_tree()
        leaf = next(
            node for node in tree.nodes.values() if not node.children_ids
        )
        with self.assertRaisesRegex(ValueError, "leaf nodes require content"):
            self._revalidate_with_nodes(
                tree, leaf.model_copy(update={"content": None})
            )

    def test_rejects_internal_node_carrying_content(self) -> None:
        tree = build_paper_structure_tree()
        root = tree.root()
        with self.assertRaisesRegex(
            ValueError, "internal nodes must not carry content"
        ):
            self._revalidate_with_nodes(
                tree, root.model_copy(update={"content": "leaked body text"})
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
