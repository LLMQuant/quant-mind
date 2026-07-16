"""Store and search one canonical knowledge item locally."""

import asyncio
from datetime import datetime, timezone

from quantmind.knowledge import News, SourceRef
from quantmind.library import LocalKnowledgeLibrary, SemanticQuery


async def main() -> None:
    """Persist one news item and print matching semantic evidence."""
    published_at = datetime(2026, 7, 16, 8, tzinfo=timezone.utc)
    item = News(
        as_of=published_at,
        available_at=published_at,
        source=SourceRef(kind="rss", uri="https://example.com/fed-rates"),
        headline="Federal Reserve holds rates steady",
        event_type="monetary_policy",
        entities=["Federal Reserve"],
        tags=["macro", "rates"],
    )
    library = await LocalKnowledgeLibrary.open(
        ".quantmind/library.db",
        embedding_model="text-embedding-3-small",
    )
    try:
        await library.put(item)
        hits = await library.search(
            SemanticQuery(
                text="central bank interest-rate decision",
                available_at_before=datetime.now(timezone.utc),
                top_k=3,
            )
        )
        for hit in hits:
            print(hit.score, hit.matched_text)
    finally:
        await library.close()


if __name__ == "__main__":
    asyncio.run(main())
