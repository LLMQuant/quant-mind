from typing import Dict, List, Optional

import networkx as nx
from camel.embeddings import BaseEmbedding

from autoscholar.knowledge.knowledge_graph import KnowledgeGraph
from autoscholar.knowledge.paper import Paper
from autoscholar.utils.logger import setup_logger
from autoscholar.utils.similarity import (
    compute_similarity_matrix,
    filter_connections_by_threshold,
)

logger = setup_logger(__name__)


class KnowledgeGraphBuilder:
    """Class for building and analyzing knowledge graphs of academic papers."""

    def __init__(self, embedding_model: Optional[BaseEmbedding] = None):
        """Initialize knowledge graph builder.

        Parameters:
        ----------
            embedding_model: BaseEmbedding instance (will create a new one if None)
        """
        self.embedding_model = embedding_model
        self.graph: Optional[KnowledgeGraph] = None
        self.papers: Optional[List[Paper]] = None
        self.embeddings = None
        self.similarity_matrix = None

    def build_graph(
        self,
        papers: List[Paper],
        embedding_dict: Optional[Dict[str, List[float]]] = None,
        similarity_threshold: float = 0.5,
    ) -> KnowledgeGraph:
        """Build a knowledge graph from paper data.

        Parameters:
        ----------
            papers: List of paper dictionaries
            embedding_dict: Dictionary of embeddings, if provided, will skip extracting
                embeddings from papers
            similarity_threshold: Minimum similarity for creating an edge

        Returns:
        -------
            KnowledgeGraph object
        """
        # Store papers
        self.papers = papers

        # [TODO] More efficient way to generate embeddings
        # Generate embeddings
        if embedding_dict is None:
            self.embeddings = [paper.embedding for paper in papers]

            if any(embedding is None for embedding in self.embeddings):
                logger.info(
                    "Some embeddings are None. Re-generating embeddings."
                )
                texts = [paper.get_text_for_embedding() for paper in papers]
                self.embeddings = self.embedding_model.embed_batch(texts)

        else:
            logger.info("Using provided embeddings.")
            self.embeddings = [
                embedding_dict[paper.paper_id] for paper in papers
            ]

        # Compute similarity matrix
        self.similarity_matrix = compute_similarity_matrix(self.embeddings)

        # Create graph
        G = nx.Graph()

        # Add nodes
        for i, paper in enumerate(papers):
            G.add_node(
                i,
                paper_id=paper.paper_id,
                title=paper.title,
                abstract=paper.abstract,
            )

        # Get connections above threshold
        connections = filter_connections_by_threshold(
            self.similarity_matrix, threshold=similarity_threshold
        )

        # Add edges
        for i, j, similarity in connections:
            if i in G and j in G:
                G.add_edge(i, j, weight=similarity)
            else:
                print(
                    f"Warning: Attempted to add edge between non-existent nodes ({i}, {j}). Skipping."
                )

        # Store graph as KnowledgeGraph instance
        self.graph = KnowledgeGraph(G, self.papers)

        return self.graph
