"""Search a ready-to-use local bundle of financial knowledge."""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from quantmind.knowledge import TreeKnowledge
from quantmind.library import LocalKnowledgeLibrary, SemanticQuery

_BUNDLE_PATH = Path(__file__).parent / "data" / "ai_infrastructure.db"
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIMENSIONS = 1536


async def main() -> None:
    """Search and resolve evidence from the prebuilt local bundle."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    library = await LocalKnowledgeLibrary.open(
        _BUNDLE_PATH,
        embedding_model=_EMBEDDING_MODEL,
        embedding_dimensions=_EMBEDDING_DIMENSIONS,
    )
    try:
        query = SemanticQuery(
            text="What evidence shows demand for AI infrastructure is expanding?",
            source_kinds=["http", "arxiv"],
            tags=["ai-infrastructure"],
            available_at_before=datetime(2026, 1, 1, tzinfo=timezone.utc),
            top_k=6,
        )
        hits = await library.search(query)
        print(f"Bundle: {_BUNDLE_PATH}")
        print(f"Query: {query.text}\n")
        for rank, hit in enumerate(hits, start=1):
            item = await library.get(hit.item_id)
            print(f"{rank}. score={hit.score:.3f} type={hit.item_type}")
            if hit.node_id is not None and isinstance(item, TreeKnowledge):
                path = " > ".join(
                    node.title for node in item.find_path(hit.node_id)
                )
                print(f"   path: {path}")
                node = item.nodes[hit.node_id]
                if node.content:
                    print(f"   content: {node.content}")
            print(f"   matched: {hit.matched_text}")
            print(f"   source: {hit.source.uri}")
            print(
                f"   as_of={hit.as_of.date()} available_at={hit.available_at}"
            )
            for citation in hit.citations:
                detail = f" — {citation.quote}" if citation.quote else ""
                print(f"   citation: {citation.source_id}{detail}")
            print()
    finally:
        await library.close()


if __name__ == "__main__":
    asyncio.run(main())
