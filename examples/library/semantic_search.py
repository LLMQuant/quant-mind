"""Run a persisted semantic search over bundled financial knowledge."""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from quantmind.knowledge import (
    BaseKnowledge,
    Earnings,
    News,
    Paper,
    TreeKnowledge,
)
from quantmind.library import LocalKnowledgeLibrary, SemanticQuery

_BUNDLE_PATH = Path(__file__).parent / "data" / "ai_infrastructure.json"
_DEFAULT_LIBRARY_PATH = Path(".quantmind/ai-infrastructure.db")
_KNOWLEDGE_TYPES: dict[str, type[BaseKnowledge]] = {
    "earnings": Earnings,
    "news": News,
    "paper": Paper,
}


def _load_bundle() -> tuple[str, list[BaseKnowledge]]:
    """Validate bundled JSON as canonical QuantMind knowledge."""
    bundle = json.loads(_BUNDLE_PATH.read_text(encoding="utf-8"))
    scenario = str(bundle["scenario"])
    items: list[BaseKnowledge] = []
    for payload in bundle["items"]:
        item_type = str(payload["item_type"])
        knowledge_type = _KNOWLEDGE_TYPES.get(item_type)
        if knowledge_type is None:
            raise ValueError(f"Unsupported bundled item_type: {item_type}")
        items.append(knowledge_type.model_validate(payload))
    return scenario, items


def _target_count(items: list[BaseKnowledge]) -> int:
    """Count the exact item/root/node projections created by the library."""
    return sum(
        len(item.nodes) if isinstance(item, TreeKnowledge) else 1
        for item in items
    )


async def main() -> None:
    """Seed, reopen, search, and resolve bundled tree evidence."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    scenario, items = _load_bundle()
    database_path = Path(
        os.getenv("QUANTMIND_LIBRARY_PATH", str(_DEFAULT_LIBRARY_PATH))
    )
    embedding_model = os.getenv(
        "QUANTMIND_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    library = await LocalKnowledgeLibrary.open(
        database_path,
        embedding_model=embedding_model,
    )
    try:
        for item in items:
            await library.put(item)
    finally:
        await library.close()

    print(f"Scenario: {scenario}")
    print(
        f"Persisted {len(items)} knowledge items / {_target_count(items)} targets"
    )
    print(f"Database: {database_path}\n")

    library = await LocalKnowledgeLibrary.open(
        database_path,
        embedding_model=embedding_model,
    )
    try:
        query = SemanticQuery(
            text="What evidence shows demand for AI infrastructure is expanding?",
            source_kinds=["http", "arxiv"],
            tags=["ai-infrastructure"],
            available_at_before=datetime(2026, 1, 1, tzinfo=timezone.utc),
            top_k=5,
        )
        hits = await library.search(query)
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
