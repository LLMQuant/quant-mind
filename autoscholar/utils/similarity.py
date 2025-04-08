# scholarauto/utils/similarity.py

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_matrix(
    embeddings: np.ndarray | List[List[float]],
) -> np.ndarray:
    """Compute cosine similarity matrix between all embeddings.

    Parameters:
    ----------
        embeddings: Matrix of embeddings (n_samples, embedding_dim)

    Returns:
    -------
        Similarity matrix (n_samples, n_samples)
    """
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    return cosine_similarity(embeddings)


def get_similar_papers(
    query_embedding: np.ndarray,
    paper_embeddings: np.ndarray,
    papers: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Find papers most similar to a query embedding.

    Parameters:
    ----------
        query_embedding: Query embedding vector
        paper_embeddings: Matrix of paper embeddings
        papers: List of paper dictionaries
        top_k: Number of top results to return

    Returns:
    -------
        List of dictionaries with papers and similarity scores
    """
    # Compute similarity between query and all papers
    similarities = cosine_similarity([query_embedding], paper_embeddings)[0]

    # Get indices of top-k papers
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Create result list
    results = []
    for idx in top_indices:
        result = {"paper": papers[idx], "similarity": float(similarities[idx])}
        results.append(result)

    return results


def filter_connections_by_threshold(
    similarity_matrix: np.ndarray, threshold: float = 0.5
) -> List[Tuple[int, int, float]]:
    """Filter connections based on similarity threshold.

    Parameters:
    ----------
        similarity_matrix: Matrix of similarities
        threshold: Minimum similarity to include connection

    Returns:
    -------
        List of tuples (node_i, node_j, similarity)
    """
    connections = []
    n = similarity_matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):  # Upper triangular to avoid duplicates
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                connections.append((i, j, similarity))

    # Sort by similarity (strongest connections first)
    connections.sort(key=lambda x: x[2], reverse=True)

    return connections
