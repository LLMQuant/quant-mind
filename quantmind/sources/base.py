"""Base source interface for content acquisition."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from quantmind.models import BaseContent
from quantmind.config import BaseSourceConfig

# Generic type for content
ContentType = TypeVar("ContentType", bound=BaseContent)


class BaseSource(Generic[ContentType], ABC):
    """Abstract base class for content sources.

    Defines the interface for acquiring content from various sources
    such as arXiv, news feeds, financial blogs, etc.

    Generic type parameter allows different sources to work with
    different content types while maintaining type safety.
    """

    def __init__(self, config: Optional[BaseSourceConfig] = None):
        """Initialize source with configuration.

        Args:
            config: Source-specific configuration
        """
        self.config = config or BaseSourceConfig()
        self.name = self.__class__.__name__.lower().replace("source", "")

    @abstractmethod
    def search(
        self, query: str, max_results: Optional[int] = None
    ) -> List[ContentType]:
        """Search for content matching the query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return, defaults to config.max_results

        Returns:
            List of content objects
        """
        pass

    @abstractmethod
    def get_by_id(self, content_id: str) -> Optional[ContentType]:
        """Get specific content by its ID.

        Args:
            content_id: Unique identifier for the content

        Returns:
            Content object if found, None otherwise
        """
        pass

    def get_batch(self, content_ids: List[str]) -> List[ContentType]:
        """Get multiple content items by their IDs.

        Args:
            content_ids: List of content identifiers

        Returns:
            List of content objects (may be shorter than input if some not found)
        """
        items = []
        for content_id in content_ids:
            item = self.get_by_id(content_id)
            if item:
                items.append(item)
        return items

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
