from typing import List

import networkx as nx

from autoscholar.knowledge.paper import Paper


class KnowledgeGraph:
    """Encapsulates the knowledge graph structure and operations."""

    def __init__(self, graph: nx.Graph, papers: List[Paper]):
        """Initialize the KnowledgeGraph.

        Parameters:
        ----------
            graph: The networkx graph object representing the knowledge graph.
            papers: The list of paper dictionaries used to build the graph.
        """
        self._graph = graph
        self._papers = papers

    @property
    def graph(self) -> nx.Graph:
        """Returns the underlying networkx graph object."""
        return self._graph

    @property
    def papers(self) -> List[Paper]:
        """Returns the list of paper dictionaries."""
        return self._papers
