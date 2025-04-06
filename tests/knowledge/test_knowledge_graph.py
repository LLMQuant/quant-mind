import unittest

import networkx as nx

from autoscholar.knowledge.knowledge_graph import KnowledgeGraph
from autoscholar.knowledge.paper import Paper


class TestKnowledgeGraph(unittest.TestCase):
    """Test the KnowledgeGraph class."""

    def setUp(self):
        """Setup test data."""
        # Create test papers
        self.paper1 = Paper(
            title="Test paper 1",
            abstract="This is test abstract 1",
            paper_id="paper1",
        )
        self.paper2 = Paper(
            title="Test paper 2",
            abstract="This is test abstract 2",
            paper_id="paper2",
        )
        self.papers = [self.paper1, self.paper2]

        # Create a test graph
        self.graph = nx.Graph()
        self.graph.add_node(self.paper1.paper_id, type="paper")
        self.graph.add_node(self.paper2.paper_id, type="paper")
        self.graph.add_edge(
            self.paper1.paper_id, self.paper2.paper_id, weight=0.8
        )

    def test_knowledge_graph_creation(self):
        """Test the creation of a knowledge graph object."""
        kg = KnowledgeGraph(graph=self.graph, papers=self.papers)

        # Test properties
        self.assertEqual(kg.graph, self.graph)
        self.assertEqual(kg.papers, self.papers)

        # Test graph structure
        self.assertEqual(len(kg.graph.nodes), 2)
        self.assertEqual(len(kg.graph.edges), 1)
        self.assertTrue(
            kg.graph.has_edge(self.paper1.paper_id, self.paper2.paper_id)
        )
        self.assertEqual(
            kg.graph[self.paper1.paper_id][self.paper2.paper_id]["weight"], 0.8
        )


if __name__ == "__main__":
    unittest.main()
