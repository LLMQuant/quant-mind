"""Compile the library example source into a ready-to-search SQLite bundle."""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from quantmind.knowledge import BaseKnowledge, Earnings, News, Paper
from quantmind.library import LocalKnowledgeLibrary

_ROOT = Path(__file__).parents[1]
_SOURCE_PATH = (
    _ROOT / "examples" / "library" / "data" / "ai_infrastructure.json"
)
_OUTPUT_PATH = _ROOT / "examples" / "library" / "data" / "ai_infrastructure.db"
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIMENSIONS = 1536
_KNOWLEDGE_TYPES: dict[str, type[BaseKnowledge]] = {
    "earnings": Earnings,
    "news": News,
    "paper": Paper,
}


def _load_source_items() -> list[BaseKnowledge]:
    """Validate the auditable source JSON as canonical knowledge."""
    bundle = json.loads(_SOURCE_PATH.read_text(encoding="utf-8"))
    items: list[BaseKnowledge] = []
    for payload in bundle["items"]:
        item_type = str(payload["item_type"])
        knowledge_type = _KNOWLEDGE_TYPES.get(item_type)
        if knowledge_type is None:
            raise ValueError(f"Unsupported source item_type: {item_type}")
        items.append(knowledge_type.model_validate(payload))
    return items


async def main() -> None:
    """Build a fresh model-specific database from the source bundle."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before building the bundle.")

    items = _load_source_items()
    _OUTPUT_PATH.unlink(missing_ok=True)
    succeeded = False
    library = await LocalKnowledgeLibrary.open(
        _OUTPUT_PATH,
        embedding_model=_EMBEDDING_MODEL,
        embedding_dimensions=_EMBEDDING_DIMENSIONS,
    )
    try:
        for item in items:
            await library.put(item)
        succeeded = True
    finally:
        await library.close()
        if not succeeded:
            _OUTPUT_PATH.unlink(missing_ok=True)

    print(
        f"Built {_OUTPUT_PATH.relative_to(_ROOT)} with {len(items)} items "
        f"using {_EMBEDDING_MODEL}."
    )


if __name__ == "__main__":
    asyncio.run(main())
