"""Base source interface for content acquisition."""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Any

from quantmind.models.paper import Paper


class BaseSource(ABC):
    """Abstract base class for content sources.

    Defines the interface for acquiring content from various sources
    such as arXiv, news feeds, financial blogs, etc.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize source with configuration.

        Args:
            config: Source-specific configuration
        """
        self.config = config or {}
        self.name = self.__class__.__name__.lower().replace("source", "")

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search for papers matching the query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of Paper objects
        """
        pass

    @abstractmethod
    def get_by_id(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by its ID.

        Args:
            paper_id: Unique identifier for the paper

        Returns:
            Paper object if found, None otherwise
        """
        pass

    @abstractmethod
    def get_recent(
        self, days: int = 7, categories: Optional[List[str]] = None
    ) -> List[Paper]:
        """Get recent papers from the source.

        Args:
            days: Number of days to look back
            categories: Optional list of categories to filter by

        Returns:
            List of recent Paper objects
        """
        pass

    def get_batch(self, paper_ids: List[str]) -> List[Paper]:
        """Get multiple papers by their IDs.

        Args:
            paper_ids: List of paper identifiers

        Returns:
            List of Paper objects (may be shorter than input if some not found)
        """
        papers = []
        for paper_id in paper_ids:
            paper = self.get_by_id(paper_id)
            if paper:
                papers.append(paper)
        return papers

    def stream_recent(
        self, days: int = 7, categories: Optional[List[str]] = None
    ) -> Iterator[Paper]:
        """Stream recent papers from the source.

        Args:
            days: Number of days to look back
            categories: Optional list of categories to filter by

        Yields:
            Paper objects one at a time
        """
        papers = self.get_recent(days=days, categories=categories)
        for paper in papers:
            yield paper

    def validate_config(self) -> bool:
        """Validate the source configuration.

        Returns:
            True if configuration is valid
        """
        return True

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about this source.

        Returns:
            Dictionary with source metadata
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
        }

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
