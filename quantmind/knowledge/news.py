"""News knowledge schema (output of news_flow)."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from quantmind.knowledge._flatten import FlattenKnowledge


class News(FlattenKnowledge):
    """A single news event extraction."""

    item_type: Literal["news"] = "news"

    headline: str
    event_type: str
    timestamp: datetime
    entities: list[str] = Field(default_factory=list)
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"
    materiality: Literal["low", "medium", "high"] = "medium"

    def embedding_text(self) -> str:
        entities = ", ".join(self.entities)
        return f"{self.headline}\n{self.event_type}\n{entities}".strip()
