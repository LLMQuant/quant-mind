"""Tests for knowledge._tree — TreeNode + TreeKnowledge."""

import unittest
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import ValidationError

from quantmind.knowledge._base import SourceRef
from quantmind.knowledge._tree import TreeKnowledge, TreeNode


def _now() -> datetime:
    return datetime(2026, 4, 26, tzinfo=timezone.utc)


def _src() -> SourceRef:
    return SourceRef(kind="manual")


class _SampleTree(TreeKnowledge):
    item_type: Literal["sample_tree"] = "sample_tree"  # pyright: ignore[reportIncompatibleVariableOverride]


def _make_tree() -> _SampleTree:
    """Three-level tree:

    root
    ├── a
    │   ├── a1
    │   └── a2
    └── b
    """
    a1_id, a2_id = uuid4(), uuid4()
    a_id = uuid4()
    b_id = uuid4()
    root_id = uuid4()
    a1 = TreeNode(
        node_id=a1_id,
        parent_id=a_id,
        position=0,
        title="A1",
        summary="leaf a1",
    )
    a2 = TreeNode(
        node_id=a2_id,
        parent_id=a_id,
        position=1,
        title="A2",
        summary="leaf a2",
    )
    a = TreeNode(
        node_id=a_id,
        parent_id=root_id,
        position=0,
        title="A",
        summary="branch a",
        children_ids=[a1_id, a2_id],
    )
    b = TreeNode(
        node_id=b_id,
        parent_id=root_id,
        position=1,
        title="B",
        summary="leaf b",
    )
    root = TreeNode(
        node_id=root_id,
        parent_id=None,
        position=0,
        title="Root",
        summary="root summary",
        children_ids=[a_id, b_id],
    )
    return _SampleTree(
        as_of=_now(),
        source=_src(),
        root_node_id=root_id,
        nodes={n.node_id: n for n in [root, a, b, a1, a2]},
    )


class TreeNodeTests(unittest.TestCase):
    def test_minimal(self):
        n = TreeNode(title="t", summary="s")
        self.assertEqual(n.title, "t")
        self.assertEqual(n.summary, "s")
        self.assertIsNone(n.content)
        self.assertEqual(n.children_ids, [])
        self.assertIsInstance(n.node_id, UUID)

    def test_default_node_id_unique(self):
        a = TreeNode(title="t", summary="s")
        b = TreeNode(title="t", summary="s")
        self.assertNotEqual(a.node_id, b.node_id)

    def test_extra_forbidden(self):
        with self.assertRaises(ValidationError):
            TreeNode(title="t", summary="s", garbage=1)  # type: ignore[call-arg]

    def test_frozen(self):
        n = TreeNode(title="t", summary="s")
        with self.assertRaises(ValidationError):
            n.title = "z"  # type: ignore[misc]

    def test_default_embedding_text(self):
        n = TreeNode(title="Methodology", summary="We use X to do Y.")
        self.assertEqual(n.embedding_text(), "Methodology\nWe use X to do Y.")


class TreeKnowledgeTests(unittest.TestCase):
    def test_root_helper(self):
        tree = _make_tree()
        root = tree.root()
        self.assertEqual(root.title, "Root")
        self.assertIsNone(root.parent_id)

    def test_children_of(self):
        tree = _make_tree()
        children = tree.children_of(tree.root_node_id)
        titles = [c.title for c in children]
        self.assertEqual(titles, ["A", "B"])

    def test_walk_dfs_order(self):
        tree = _make_tree()
        order = [n.title for n in tree.walk_dfs()]
        self.assertEqual(order, ["Root", "A", "A1", "A2", "B"])

    def test_find_path_root(self):
        tree = _make_tree()
        path = tree.find_path(tree.root_node_id)
        self.assertEqual([n.title for n in path], ["Root"])

    def test_find_path_leaf(self):
        tree = _make_tree()
        # Find a1 from the tree's flat node map.
        a1 = next(n for n in tree.nodes.values() if n.title == "A1")
        path = tree.find_path(a1.node_id)
        self.assertEqual([n.title for n in path], ["Root", "A", "A1"])

    def test_find_path_unknown(self):
        tree = _make_tree()
        self.assertEqual(tree.find_path(uuid4()), [])

    def test_embedding_text_uses_root(self):
        tree = _make_tree()
        self.assertEqual(tree.embedding_text(), "Root\nroot summary")


if __name__ == "__main__":
    unittest.main()
