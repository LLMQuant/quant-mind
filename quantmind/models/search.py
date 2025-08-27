"""Search content model."""

from typing import Any, Dict, List, Optional

from quantmind.models.content import BaseContent


class SearchContent(BaseContent):
    """Represents content from a search engine result."""

    title: str
    url: str
    snippet: str
    source: str = "search"
    query: Optional[str] = None
    meta_info: Dict[str, Any] = {}

    def get_primary_id(self) -> str:
        """Return the primary identifier for the content."""
        return self.url

    def get_text_for_embedding(self) -> str:
        """Return the text to be used for generating embeddings."""
        return f"{self.title}{self.snippet}"

    def to_dict(self) -> Dict[str, any]:
        """Convert the object to a dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "query": self.query,
            "meta_info": self.meta_info,
        }
