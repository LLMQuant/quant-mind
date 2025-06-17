"""Base parser interface for content extraction."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

from quantmind.models.paper import Paper


class BaseParser(ABC):
    """Abstract base class for content parsers.

    Defines the interface for extracting and processing content
    from various document formats (PDF, HTML, etc.).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize parser with configuration.

        Args:
            config: Parser-specific configuration
        """
        self.config = config or {}
        self.name = self.__class__.__name__.lower().replace("parser", "")

    @abstractmethod
    def parse_paper(self, paper: Paper) -> Paper:
        """Parse and enrich paper content.

        Args:
            paper: Paper object to parse

        Returns:
            Paper object with enriched content
        """
        pass

    @abstractmethod
    def parse_content(self, content: str, content_type: str = "text") -> str:
        """Parse raw content into structured text.

        Args:
            content: Raw content to parse
            content_type: Type of content (pdf, html, text, etc.)

        Returns:
            Parsed and cleaned text content
        """
        pass

    def extract_from_url(self, url: str) -> str:
        """Extract content from a URL.

        Args:
            url: URL to extract content from

        Returns:
            Extracted text content

        Raises:
            NotImplementedError: If URL extraction not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support URL extraction"
        )

    def extract_from_file(self, file_path: str) -> str:
        """Extract content from a local file.

        Args:
            file_path: Path to file to extract content from

        Returns:
            Extracted text content

        Raises:
            NotImplementedError: If file extraction not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support file extraction"
        )

    def validate_content(self, content: str) -> bool:
        """Validate extracted content quality.

        Args:
            content: Content to validate

        Returns:
            True if content meets quality standards
        """
        if not content or not isinstance(content, str):
            return False

        # Basic quality checks
        if len(content.strip()) < 100:  # Too short
            return False

        # Check for reasonable word count
        words = content.split()
        if len(words) < 20:  # Too few words
            return False

        # Check for excessive repeated characters (parsing errors)
        for char in ["\n", " ", "\t"]:
            if char * 10 in content:  # 10+ consecutive same characters
                return False

        return True

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        import re

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Remove common PDF artifacts
        text = re.sub(r"\x0c", "", text)  # Form feed characters
        text = re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text
        )  # Control characters

        # Fix common encoding issues
        replacements = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            """: "'",
            """: "'",
            '"': '"',
            '"': '"',
            "–": "-",
            "—": "-",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text.strip()

    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about this parser.

        Returns:
            Dictionary with parser metadata
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
        }

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
