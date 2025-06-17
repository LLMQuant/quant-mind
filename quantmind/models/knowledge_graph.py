"""Enhanced knowledge graph model for QuantMind."""

from typing import Dict, List, Optional, Any, Tuple, Set
import networkx as nx
import numpy as np
from datetime import datetime

from quantmind.models.paper import Paper
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeGraph:
    """Enhanced knowledge graph for representing paper relationships.

    Builds on NetworkX to provide a rich graph structure with support for
    multiple node and edge types, metadata, and advanced graph operations.
    """

    def __init__(self, directed: bool = False):
        """Initialize knowledge graph.

        Args:
            directed: Whether to create a directed graph
        """
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.papers: Dict[str, Paper] = {}
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.utcnow(),
            "node_types": set(),
            "edge_types": set(),
            "statistics": {},
        }

        logger.debug("KnowledgeGraph initialized")

    def add_paper(self, paper: Paper, node_type: str = "paper") -> str:
        """Add a paper as a node in the graph.

        Args:
            paper: Paper object to add
            node_type: Type of node (default: "paper")

        Returns:
            Node ID in the graph
        """
        node_id = paper.get_primary_id()

        # Store paper
        self.papers[node_id] = paper

        # Add node with attributes
        self.graph.add_node(
            node_id,
            node_type=node_type,
            title=paper.title,
            abstract=paper.abstract[:200],  # Shortened for graph storage
            categories=paper.categories,
            tags=paper.tags,
            published_date=paper.published_date,
            authors=paper.authors,
            paper_object=paper,
        )

        # Update metadata
        self.metadata["node_types"].add(node_type)
        self._update_statistics()

        logger.debug(f"Added paper node: {node_id}")
        return node_id

    def add_papers(self, papers: List[Paper]) -> List[str]:
        """Add multiple papers to the graph.

        Args:
            papers: List of Paper objects

        Returns:
            List of node IDs
        """
        return [self.add_paper(paper) for paper in papers]

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "related",
        weight: float = 1.0,
        **attributes,
    ) -> None:
        """Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge
            weight: Edge weight
            **attributes: Additional edge attributes
        """
        if source_id not in self.graph or target_id not in self.graph:
            raise ValueError("Both nodes must exist in the graph")

        self.graph.add_edge(
            source_id,
            target_id,
            edge_type=edge_type,
            weight=weight,
            created_at=datetime.utcnow(),
            **attributes,
        )

        # Update metadata
        self.metadata["edge_types"].add(edge_type)
        self._update_statistics()

        logger.debug(f"Added edge: {source_id} -> {target_id} ({edge_type})")

    def connect_similar_papers(
        self, similarity_threshold: float = 0.7, method: str = "embedding"
    ) -> int:
        """Connect papers based on similarity.

        Args:
            similarity_threshold: Minimum similarity for connection
            method: Similarity computation method

        Returns:
            Number of edges added
        """
        if method == "embedding":
            return self._connect_by_embedding_similarity(similarity_threshold)
        elif method == "category":
            return self._connect_by_category_similarity(similarity_threshold)
        elif method == "keyword":
            return self._connect_by_keyword_similarity(similarity_threshold)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def connect_by_authors(self) -> int:
        """Connect papers by shared authors.

        Returns:
            Number of edges added
        """
        edges_added = 0
        paper_nodes = [
            n
            for n in self.graph.nodes()
            if self.graph.nodes[n]["node_type"] == "paper"
        ]

        for i, node1 in enumerate(paper_nodes):
            for node2 in paper_nodes[i + 1 :]:
                paper1 = self.papers[node1]
                paper2 = self.papers[node2]

                shared_authors = set(paper1.authors) & set(paper2.authors)
                if shared_authors:
                    self.add_edge(
                        node1,
                        node2,
                        edge_type="shared_author",
                        weight=len(shared_authors)
                        / max(len(paper1.authors), len(paper2.authors)),
                        shared_authors=list(shared_authors),
                    )
                    edges_added += 1

        logger.info(f"Connected {edges_added} paper pairs by shared authors")
        return edges_added

    def connect_by_categories(self) -> int:
        """Connect papers by shared categories.

        Returns:
            Number of edges added
        """
        edges_added = 0
        paper_nodes = [
            n
            for n in self.graph.nodes()
            if self.graph.nodes[n]["node_type"] == "paper"
        ]

        for i, node1 in enumerate(paper_nodes):
            for node2 in paper_nodes[i + 1 :]:
                paper1 = self.papers[node1]
                paper2 = self.papers[node2]

                shared_categories = set(paper1.categories) & set(
                    paper2.categories
                )
                if shared_categories:
                    self.add_edge(
                        node1,
                        node2,
                        edge_type="shared_category",
                        weight=len(shared_categories)
                        / max(len(paper1.categories), len(paper2.categories)),
                        shared_categories=list(shared_categories),
                    )
                    edges_added += 1

        logger.info(f"Connected {edges_added} paper pairs by shared categories")
        return edges_added

    def get_paper(self, node_id: str) -> Optional[Paper]:
        """Get paper by node ID.

        Args:
            node_id: Node identifier

        Returns:
            Paper object if found
        """
        return self.papers.get(node_id)

    def get_papers_by_category(self, category: str) -> List[Paper]:
        """Get papers in a specific category.

        Args:
            category: Category name

        Returns:
            List of papers in the category
        """
        papers = []
        for node_id, paper in self.papers.items():
            if category in paper.categories:
                papers.append(paper)
        return papers

    def get_related_papers(
        self,
        node_id: str,
        max_papers: int = 10,
        edge_types: Optional[List[str]] = None,
    ) -> List[Tuple[Paper, float]]:
        """Get papers related to a given paper.

        Args:
            node_id: Paper node ID
            max_papers: Maximum number of related papers
            edge_types: Optional list of edge types to consider

        Returns:
            List of (paper, weight) tuples sorted by weight
        """
        if node_id not in self.graph:
            return []

        related = []
        for neighbor in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)

            # Filter by edge type if specified
            if edge_types and edge_data.get("edge_type") not in edge_types:
                continue

            weight = edge_data.get("weight", 1.0)
            paper = self.papers.get(neighbor)
            if paper:
                related.append((paper, weight))

        # Sort by weight (descending) and limit
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:max_papers]

    def get_central_papers(
        self, metric: str = "degree", top_k: int = 10
    ) -> List[Tuple[Paper, float]]:
        """Get most central papers in the graph.

        Args:
            metric: Centrality metric ("degree", "betweenness", "closeness", "pagerank")
            top_k: Number of top papers to return

        Returns:
            List of (paper, centrality_score) tuples
        """
        if metric == "degree":
            centrality = nx.degree_centrality(self.graph)
        elif metric == "betweenness":
            centrality = nx.betweenness_centrality(self.graph)
        elif metric == "closeness":
            centrality = nx.closeness_centrality(self.graph)
        elif metric == "pagerank":
            centrality = nx.pagerank(self.graph)
        else:
            raise ValueError(f"Unknown centrality metric: {metric}")

        # Get top papers
        sorted_nodes = sorted(
            centrality.items(), key=lambda x: x[1], reverse=True
        )
        top_papers = []

        for node_id, score in sorted_nodes[:top_k]:
            paper = self.papers.get(node_id)
            if paper:
                top_papers.append((paper, score))

        return top_papers

    def find_communities(
        self, algorithm: str = "louvain"
    ) -> Dict[int, List[str]]:
        """Detect communities in the graph.

        Args:
            algorithm: Community detection algorithm

        Returns:
            Dictionary mapping community ID to list of node IDs
        """
        if algorithm == "louvain":
            import networkx.algorithms.community as nx_comm

            communities = nx_comm.louvain_communities(self.graph)
        else:
            raise ValueError(
                f"Unknown community detection algorithm: {algorithm}"
            )

        # Convert to dictionary format
        community_dict = {}
        for i, community in enumerate(communities):
            community_dict[i] = list(community)

        logger.info(
            f"Found {len(community_dict)} communities using {algorithm}"
        )
        return community_dict

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics.

        Returns:
            Dictionary with graph statistics
        """
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "papers": len(self.papers),
            "node_types": list(self.metadata["node_types"]),
            "edge_types": list(self.metadata["edge_types"]),
            "is_connected": nx.is_connected(self.graph)
            if not self.graph.is_directed()
            else nx.is_weakly_connected(self.graph),
            "density": nx.density(self.graph),
        }

        if stats["nodes"] > 0:
            stats["average_degree"] = 2 * stats["edges"] / stats["nodes"]

        # Category distribution
        category_counts = {}
        for paper in self.papers.values():
            for category in paper.categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        stats["category_distribution"] = category_counts

        return stats

    def export_for_visualization(
        self, layout: str = "spring", include_attributes: bool = True
    ) -> Dict[str, Any]:
        """Export graph data for visualization.

        Args:
            layout: Layout algorithm for node positioning
            include_attributes: Whether to include node/edge attributes

        Returns:
            Dictionary with nodes, edges, and layout information
        """
        # Calculate layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "random":
            pos = nx.random_layout(self.graph)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # Prepare nodes
        nodes = []
        for node_id in self.graph.nodes():
            node_data = {
                "id": node_id,
                "x": pos[node_id][0],
                "y": pos[node_id][1],
            }

            if include_attributes:
                node_data.update(self.graph.nodes[node_id])
                # Convert datetime objects to strings
                if (
                    "published_date" in node_data
                    and node_data["published_date"]
                ):
                    node_data["published_date"] = node_data[
                        "published_date"
                    ].isoformat()

            nodes.append(node_data)

        # Prepare edges
        edges = []
        for source, target in self.graph.edges():
            edge_data = {"source": source, "target": target}

            if include_attributes:
                edge_attrs = self.graph.get_edge_data(source, target)
                edge_data.update(edge_attrs)
                # Convert datetime objects to strings
                if "created_at" in edge_data and edge_data["created_at"]:
                    edge_data["created_at"] = edge_data[
                        "created_at"
                    ].isoformat()

            edges.append(edge_data)

        return {
            "nodes": nodes,
            "edges": edges,
            "layout": layout,
            "statistics": self.get_graph_statistics(),
        }

    def _connect_by_embedding_similarity(self, threshold: float) -> int:
        """Connect papers based on embedding similarity."""
        edges_added = 0
        paper_nodes = [
            n
            for n in self.graph.nodes()
            if self.graph.nodes[n]["node_type"] == "paper"
        ]

        # Get papers with embeddings
        papers_with_embeddings = []
        for node_id in paper_nodes:
            paper = self.papers[node_id]
            if paper.embedding:
                papers_with_embeddings.append((node_id, paper))

        if len(papers_with_embeddings) < 2:
            logger.warning(
                "Not enough papers with embeddings for similarity connections"
            )
            return 0

        # Calculate pairwise similarities
        for i, (node1, paper1) in enumerate(papers_with_embeddings):
            for node2, paper2 in papers_with_embeddings[i + 1 :]:
                similarity = self._cosine_similarity(
                    paper1.embedding, paper2.embedding
                )

                if similarity >= threshold:
                    self.add_edge(
                        node1,
                        node2,
                        edge_type="embedding_similarity",
                        weight=similarity,
                        similarity_score=similarity,
                    )
                    edges_added += 1

        logger.info(
            f"Connected {edges_added} paper pairs by embedding similarity"
        )
        return edges_added

    def _connect_by_category_similarity(self, threshold: float) -> int:
        """Connect papers based on category overlap."""
        edges_added = 0
        paper_nodes = [
            n
            for n in self.graph.nodes()
            if self.graph.nodes[n]["node_type"] == "paper"
        ]

        for i, node1 in enumerate(paper_nodes):
            for node2 in paper_nodes[i + 1 :]:
                paper1 = self.papers[node1]
                paper2 = self.papers[node2]

                if not paper1.categories or not paper2.categories:
                    continue

                # Jaccard similarity
                set1 = set(paper1.categories)
                set2 = set(paper2.categories)
                similarity = len(set1 & set2) / len(set1 | set2)

                if similarity >= threshold:
                    self.add_edge(
                        node1,
                        node2,
                        edge_type="category_similarity",
                        weight=similarity,
                        similarity_score=similarity,
                    )
                    edges_added += 1

        logger.info(
            f"Connected {edges_added} paper pairs by category similarity"
        )
        return edges_added

    def _connect_by_keyword_similarity(self, threshold: float) -> int:
        """Connect papers based on keyword overlap in title/abstract."""
        edges_added = 0
        paper_nodes = [
            n
            for n in self.graph.nodes()
            if self.graph.nodes[n]["node_type"] == "paper"
        ]

        # Extract keywords from each paper
        paper_keywords = {}
        for node_id in paper_nodes:
            paper = self.papers[node_id]
            keywords = self._extract_keywords(f"{paper.title} {paper.abstract}")
            paper_keywords[node_id] = keywords

        # Calculate pairwise similarities
        for i, node1 in enumerate(paper_nodes):
            for node2 in paper_nodes[i + 1 :]:
                keywords1 = paper_keywords[node1]
                keywords2 = paper_keywords[node2]

                if not keywords1 or not keywords2:
                    continue

                # Jaccard similarity
                similarity = len(keywords1 & keywords2) / len(
                    keywords1 | keywords2
                )

                if similarity >= threshold:
                    self.add_edge(
                        node1,
                        node2,
                        edge_type="keyword_similarity",
                        weight=similarity,
                        similarity_score=similarity,
                    )
                    edges_added += 1

        logger.info(
            f"Connected {edges_added} paper pairs by keyword similarity"
        )
        return edges_added

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        import re

        # Simple keyword extraction (can be enhanced)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Remove common stop words
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "man",
            "new",
            "now",
            "old",
            "see",
            "two",
            "way",
            "who",
            "boy",
            "did",
            "its",
            "let",
            "put",
            "say",
            "she",
            "too",
            "use",
            "this",
            "that",
            "with",
        }

        keywords = {
            word for word in words if word not in stop_words and len(word) > 3
        }
        return keywords

    def _update_statistics(self):
        """Update internal statistics."""
        self.metadata["statistics"] = self.get_graph_statistics()
        self.metadata["updated_at"] = datetime.utcnow()

    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return self.graph.number_of_nodes()

    def __str__(self) -> str:
        """String representation."""
        return f"KnowledgeGraph({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)"
