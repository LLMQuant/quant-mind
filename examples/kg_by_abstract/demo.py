import json
import os
from pathlib import Path

from camel.embeddings import OpenAIEmbedding
from camel.types import EmbeddingModelType

from autoscholar.knowledge import KnowledgeGraphBuilder, Paper
from autoscholar.visualization.graph_visualizer import GraphVisualizer

EMBEDDING_PATH = Path("examples/kg_by_abstract/example-data/embeddings")
PAPER_JSON_PATH = Path("examples/kg_by_abstract/example-data")


def main():
    """Main function."""
    json_paths = list(PAPER_JSON_PATH.glob("*.json"))
    papers = Paper.load_paper_from_paths(json_paths)
    print(papers, len(papers))

    embedding_dict = {}
    embedding_file = EMBEDDING_PATH / "embeddings.json"

    # Check if the embedding file exists
    if os.path.exists(embedding_file):
        print(f"Loading embeddings from {embedding_file}")
        with open(embedding_file, "r") as f:
            full_embedding_dict = json.load(f)
        embedding_dict = {
            paper.paper_id: full_embedding_dict[paper.paper_id]
            for paper in papers
        }
    else:
        print(
            f"Embedding file not found. Generating embeddings using OpenAI..."
        )
        # Create OpenAI embedding model
        openai_embedding = OpenAIEmbedding(
            model_type=EmbeddingModelType.TEXT_EMBEDDING_3_SMALL
        )

        # Generate embeddings for each paper
        for paper in papers:
            text = paper.get_text_for_embedding()
            embedding = openai_embedding.embed(text)
            embedding_dict[paper.paper_id] = embedding

        # Ensure embeddings directory exists
        os.makedirs(EMBEDDING_PATH, exist_ok=True)

        # Save embeddings to file
        with open(embedding_file, "w") as f:
            json.dump(embedding_dict, f)
        print(f"Embeddings saved to {embedding_file}")

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
