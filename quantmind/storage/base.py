"""Base storage interface for QuantMind knowledge base."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Iterator

from quantmind.models.paper import Paper


class BaseStorage(ABC):
    """Abstract base class for knowledge storage backends.

    Defines the interface for storing and retrieving papers and other
    knowledge entities in the QuantMind system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize storage with configuration.

        Args:
            config: Storage-specific configuration
        """
        self.config = config or {}
        self.name = self.__class__.__name__.lower().replace("storage", "")

    @abstractmethod
    def store_paper(self, paper: Paper) -> str:
        """Store a paper in the knowledge base.

        Args:
            paper: Paper object to store

        Returns:
            Unique identifier for the stored paper
        """
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Retrieve a paper by its ID.

        Args:
            paper_id: Unique identifier of the paper

        Returns:
            Paper object if found, None otherwise
        """
        pass

    @abstractmethod
    def update_paper(self, paper: Paper) -> bool:
        """Update an existing paper.

        Args:
            paper: Paper object with updated data

        Returns:
            True if update was successful
        """
        pass

    @abstractmethod
    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper from storage.

        Args:
            paper_id: Unique identifier of the paper to delete

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    def search_papers(
        self,
        query: Optional[str] = None,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Paper]:
        """Search for papers matching criteria.

        Args:
            query: Text query to search in title/abstract
            categories: List of categories to filter by
            tags: List of tags to filter by
            limit: Maximum number of results

        Returns:
            List of matching Paper objects
        """
        pass

    @abstractmethod
    def get_all_papers(self) -> Iterator[Paper]:
        """Get all papers in storage.

        Returns:
            Iterator over all Paper objects
        """
        pass

    def store_papers(self, papers: List[Paper]) -> List[str]:
        """Store multiple papers.

        Args:
            papers: List of Paper objects to store

        Returns:
            List of unique identifiers for stored papers
        """
        return [self.store_paper(paper) for paper in papers]

    def get_papers(self, paper_ids: List[str]) -> List[Paper]:
        """Retrieve multiple papers by their IDs.

        Args:
            paper_ids: List of paper identifiers

        Returns:
            List of Paper objects (may be shorter if some not found)
        """
        papers = []
        for paper_id in paper_ids:
            paper = self.get_paper(paper_id)
            if paper:
                papers.append(paper)
        return papers

    def count_papers(
        self,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """Count papers matching criteria.

        Args:
            categories: Optional list of categories to filter by
            tags: Optional list of tags to filter by

        Returns:
            Number of matching papers
        """
        # Default implementation - may be overridden for efficiency
        papers = self.search_papers(
            categories=categories, tags=tags, limit=None
        )
        return len(papers)

    def get_categories(self) -> List[str]:
        """Get all unique categories in storage.

        Returns:
            List of category strings
        """
        categories = set()
        for paper in self.get_all_papers():
            categories.update(paper.categories)
        return sorted(list(categories))

    def get_tags(self) -> List[str]:
        """Get all unique tags in storage.

        Returns:
            List of tag strings
        """
        tags = set()
        for paper in self.get_all_papers():
            tags.update(paper.tags)
        return sorted(list(tags))

    def get_papers_by_category(
        self, category: str, limit: int = 100
    ) -> List[Paper]:
        """Get papers in a specific category.

        Args:
            category: Category name
            limit: Maximum number of results

        Returns:
            List of Paper objects
        """
        return self.search_papers(categories=[category], limit=limit)

    def get_papers_by_tag(self, tag: str, limit: int = 100) -> List[Paper]:
        """Get papers with a specific tag.

        Args:
            tag: Tag name
            limit: Maximum number of results

        Returns:
            List of Paper objects
        """
        return self.search_papers(tags=[tag], limit=limit)

    def paper_exists(self, paper_id: str) -> bool:
        """Check if a paper exists in storage.

        Args:
            paper_id: Paper identifier to check

        Returns:
            True if paper exists
        """
        return self.get_paper(paper_id) is not None

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about this storage backend.

        Returns:
            Dictionary with storage metadata
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
            "paper_count": self.count_papers(),
        }

    def validate_connection(self) -> bool:
        """Validate storage connection and configuration.

        Returns:
            True if storage is accessible and properly configured
        """
        try:
            # Try a simple operation
            self.count_papers()
            return True
        except Exception:
            return False

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
