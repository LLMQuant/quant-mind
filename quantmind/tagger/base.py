"""Base tagger interface for content classification."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from quantmind.models.paper import Paper


class BaseTagger(ABC):
    """Abstract base class for content tagging and classification.

    Defines the interface for extracting tags, categories, and classifications
    from research papers and other content.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize tagger with configuration.

        Args:
            config: Tagger-specific configuration
        """
        self.config = config or {}
        self.name = self.__class__.__name__.lower().replace("tagger", "")

    @abstractmethod
    def tag_paper(self, paper: Paper) -> Paper:
        """Add tags and categories to a paper.

        Args:
            paper: Paper object to tag

        Returns:
            Paper object with added tags and categories
        """
        pass

    def tag_papers(self, papers: List[Paper]) -> List[Paper]:
        """Tag multiple papers.

        Args:
            papers: List of Paper objects to tag

        Returns:
            List of tagged Paper objects
        """
        return [self.tag_paper(paper) for paper in papers]

    @abstractmethod
    def extract_categories(self, text: str, title: str = "") -> List[str]:
        """Extract categories from text content.

        Args:
            text: Text content to analyze
            title: Optional title for additional context

        Returns:
            List of category strings
        """
        pass

    @abstractmethod
    def extract_tags(self, text: str, title: str = "") -> List[str]:
        """Extract tags from text content.

        Args:
            text: Text content to analyze
            title: Optional title for additional context

        Returns:
            List of tag strings
        """
        pass

    def get_confidence_score(self, paper: Paper) -> float:
        """Get confidence score for the tagging results.

        Args:
            paper: Tagged paper object

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Default implementation returns neutral confidence
        return 0.5

    def validate_tags(self, tags: List[str]) -> List[str]:
        """Validate and clean extracted tags.

        Args:
            tags: List of raw tags

        Returns:
            List of validated and cleaned tags
        """
        valid_tags = []
        for tag in tags:
            if isinstance(tag, str) and len(tag.strip()) > 0:
                cleaned_tag = tag.strip().lower()
                if cleaned_tag not in valid_tags:
                    valid_tags.append(cleaned_tag)
        return valid_tags

    def validate_categories(self, categories: List[str]) -> List[str]:
        """Validate and clean extracted categories.

        Args:
            categories: List of raw categories

        Returns:
            List of validated and cleaned categories
        """
        return self.validate_tags(categories)  # Same validation logic

    def get_tagger_info(self) -> Dict[str, Any]:
        """Get information about this tagger.

        Returns:
            Dictionary with tagger metadata
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
        }

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
