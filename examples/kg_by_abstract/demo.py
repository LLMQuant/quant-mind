import json
from pathlib import Path

from autoscholar.knowledge import KnowledgeGraphBuilder, Paper
from autoscholar.visualization.graph_visualizer import GraphVisualizer

EMBEDDING_PATH = Path("examples/kg_by_abstract/example-data/embeddings")
PAPER_JSON_PATH = Path("examples/kg_by_abstract/example-data")


def main():
    """Main function."""
    json_paths = list(PAPER_JSON_PATH.glob("*.json"))
    papers = Paper.load_paper_from_paths(json_paths)
    print(papers, len(papers))

    full_embedding_dict = {}
    with open(EMBEDDING_PATH / "embeddings.json", "r") as f:
        full_embedding_dict = json.load(f)

    embedding_dict = {
        paper.paper_id: full_embedding_dict[paper.paper_id] for paper in papers
    }

    builder = KnowledgeGraphBuilder()
    knowledge_graph = builder.build_graph(
        papers, embedding_dict, similarity_threshold=0.5
    )

    visualizer = GraphVisualizer(knowledge_graph)

    # Use Pyvis to generate an interactive HTML (similar to ConnectedPapers)
    html_path = visualizer.visualize_pyvis(
        output_path="paper_network.html",
        physics_settings={
            "gravity": -50000,
            "central_gravity": 0.4,
            "spring_length": 200,
        },
    )
    print(f"Interactive graph saved to: {html_path}")


if __name__ == "__main__":
    main()
