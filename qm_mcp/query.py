"""Semantic query over the corpus: embed -> cosine top-k -> grounded answer."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from qm_mcp.config import load_secrets
from qm_mcp.embed import embed_text, synthesize_answer
from qm_mcp.store import CorpusStore

_log = logging.getLogger(__name__)


async def query(
    question: str,
    *,
    k: int = 6,
    distance_threshold: float = 0.7,
    synthesize: bool = True,
    store: CorpusStore | None = None,
) -> dict[str, Any]:
    """Answer ``question`` from the corpus.

    Returns ``{question, answer, sources:[{id,title,score,source,authors}]}``.
    ``answer`` is None when ``synthesize=False`` (retrieval-only mode).
    Chunks with cosine distance > ``distance_threshold`` are filtered out;
    returns empty sources when no candidates clear the threshold (VECTOR_DISTANCE_THRESHOLD
    pattern, Ch 14 Gulli).
    """
    load_secrets()
    store = store or CorpusStore()

    if len(store) == 0:
        return {
            "question": question,
            "answer": "The corpus is empty — ingest some research first.",
            "sources": [],
        }

    q_vec = await asyncio.to_thread(embed_text, question)
    hits = store.search(q_vec, k=k)

    # store.search returns cosine similarity (higher = better).
    # cosine_distance = 1 - similarity; keep only close-enough matches.
    min_score = 1.0 - distance_threshold
    hits = [(item_id, score) for item_id, score in hits if score >= min_score]

    if not hits:
        _log.info(
            "qm_query: no candidates above threshold %.2f for question %r",
            distance_threshold,
            question[:80],
        )
        return {"question": question, "answer": None, "sources": []}

    sources: list[dict[str, Any]] = []
    contexts: list[dict[str, str]] = []
    for item_id, score in hits:
        rec = store.get(item_id)
        if not rec:
            continue
        sources.append(
            {
                "id": item_id,
                "title": rec.get("title"),
                "score": round(score, 4),
                "source_type": rec.get("source_type"),
                "source": rec.get("source"),
                "authors": rec.get("authors", []),
            }
        )
        contexts.append(
            {
                "title": rec.get("title", "untitled"),
                "source": rec.get("arxiv_id") or rec.get("source", "?"),
                "text": rec.get("full_context") or rec.get("summary", ""),
            }
        )

    answer = None
    if synthesize:
        answer = await asyncio.to_thread(synthesize_answer, question, contexts)

    return {"question": question, "answer": answer, "sources": sources}
