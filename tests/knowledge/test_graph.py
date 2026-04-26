"""Tests for knowledge._graph — placeholder semantics."""

import unittest

from quantmind.knowledge._base import BaseKnowledge
from quantmind.knowledge._graph import GraphKnowledge


class GraphKnowledgePlaceholderTests(unittest.TestCase):
    def test_class_inherits_base(self):
        # The class itself exists so users can type-hint it.
        self.assertTrue(issubclass(GraphKnowledge, BaseKnowledge))

    def test_subclassing_blocked(self):
        with self.assertRaises(NotImplementedError):

            class _AttemptedGraph(GraphKnowledge):
                pass


if __name__ == "__main__":
    unittest.main()
