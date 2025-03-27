from pathlib import Path
from typing import Any, Dict, Optional, Union

from pyvis.network import Network

from ..knowledge.knowledge_graph import KnowledgeGraph


class GraphVisualizer:
    """Visualize knowledge graphs with interactive features."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        """Initialize the visualizer with a knowledge graph.

        Parameters:
            knowledge_graph: KnowledgeGraph object containing graph and papers
        """
        self.kg = knowledge_graph
        self.graph = knowledge_graph.graph
        self.papers = knowledge_graph.papers

    def visualize_pyvis(
        self,
        output_path: Union[str, Path] = "paper_graph.html",
        width: str = "100%",
        height: str = "800px",
        physics_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create an interactive visualization using Pyvis.

        Parameters:
            output_path: Path to save the HTML output
            width: Width of the visualization
            height: Height of the visualization
            physics_settings: Custom physics settings for the network

        Returns:
            Path to the saved HTML file
        """
        # Create Pyvis network
        net = Network(height, width, notebook=False, directed=False)

        # Configure physics
        if physics_settings:
            net.barnes_hut(**physics_settings)
        else:
            net.barnes_hut(
                gravity=-80000, central_gravity=0.3, spring_length=250
            )

        # Create paper lookup for quick access
        paper_lookup = {paper.paper_id: paper for paper in self.papers}

        # Store node mapping between graph node IDs and paper IDs
        node_mapping = {}

        # First, add all nodes to ensure they exist before adding edges
        for node_id in self.graph.nodes():
            # Get node attributes
            node_data = self.graph.nodes[node_id]

            # Check if paper_id exists in the node data
            paper_id = node_data.get("paper_id")

            # If not found in node data, try to find a paper with matching id
            if not paper_id:
                # Try to find if this node_id is already a paper_id
                if node_id in paper_lookup:
                    paper_id = node_id
                else:
                    # Skip nodes that we can't map to papers
                    print(
                        f"Warning: Node {node_id} could not be mapped to a paper"
                    )
                    continue

            # Store mapping
            node_mapping[node_id] = paper_id

            # Get the paper
            paper = paper_lookup.get(paper_id)

            if not paper:
                print(f"Warning: Paper not found for ID {paper_id}")
                continue

            # Get paper metadata
            title = paper.title
            authors = paper.meta_info.get("authors", [])
            first_author = authors[0] if authors else "Unknown"
            year = paper.meta_info.get("year", "")
            citations = paper.meta_info.get("citation_count", 10)

            # Create label (shortened title)
            label = f"{title[:40]}..." if len(title) > 40 else title

            # Create tooltip with more details
            tooltip = f"<b>{title}</b><br>{first_author} et al., {year}"

            # Get node color
            color = self._get_node_color(paper)

            # Use paper_id as node id in the visualization
            net.add_node(
                paper_id, label=label, title=tooltip, size=30, color=color
            )

        # Add edges with weights, using the mapped node IDs
        for edge in self.graph.edges(data=True):
            source, target, data = edge

            # Map source and target to paper_ids
            source_paper_id = node_mapping.get(source)
            target_paper_id = node_mapping.get(target)

            # Skip edge if mapping not found
            if not source_paper_id or not target_paper_id:
                print(
                    f"Warning: Skipping edge {source}-{target} due to missing node mapping"
                )
                continue

            similarity = data.get("weight", 0.5)

            # Format similarity for display with percentage
            similarity_label = f"{similarity:.0%}"

            net.add_edge(
                source_paper_id,
                target_paper_id,
                value=similarity,
                width=similarity * 5,  # Scale for visualization
                title=f"Similarity: {similarity:.2f}",
                label=similarity_label,  # Add label to show similarity on the edge
                font={"size": 10, "color": "#555555", "align": "middle"},
                smooth={
                    "type": "curvedCW",
                    "roundness": 0.2,
                },  # Curved edges for better visibility
            )

        # Save to HTML file
        output_path = str(output_path)  # Convert Path to string if needed
        net.show(output_path)
        return output_path

    def _get_node_color(self, paper):
        """Determine node color based on paper properties.

        Parameters:
            paper: Paper object

        Returns:
            Dictionary with color configuration
        """
        year = paper.meta_info.get("year", 0)

        if not year:
            return {"background": "#97C2FC", "border": "#2B7CE9"}

        try:
            year = int(year)
            # Color gradient from light blue (older) to dark blue (newer)
            if year < 2010:
                return {"background": "#D2E5FF", "border": "#2B7CE9"}
            elif year < 2015:
                return {"background": "#AED6F1", "border": "#2B7CE9"}
            elif year < 2020:
                return {"background": "#5DADE2", "border": "#2B7CE9"}
            else:
                return {"background": "#2E86C1", "border": "#2B7CE9"}
        except:
            return {"background": "#97C2FC", "border": "#2B7CE9"}
