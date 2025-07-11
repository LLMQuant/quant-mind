"""Base storage interface for QuantMind knowledge base."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from quantmind.models import KnowledgeItem, Paper
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class BaseStorage(ABC):
    """Abstract base class for knowledge storage backends.

    Manages four types of data:
    - Raw Files: PDFs, markdown files, etc.
    - Knowledges: KnowledgeItem objects as JSON
    - Embeddings: Embedding vectors as arrays
    - Extra: Additional metadata and hashes
    """

    # Raw Files Management
    @abstractmethod
    def store_raw_file(
        self,
        file_id: str,
        file_path: Optional[Path] = None,
        content: Optional[bytes] = None,
        file_extension: str = "",
    ) -> str:
        """Store a raw file (PDF, markdown, etc.).

        Args:
            file_id: Unique identifier for the file
            file_path: Path to existing file to copy (mutually exclusive with content)
            content: Raw bytes content to write directly (mutually exclusive with file_path)
            file_extension: File extension when using content (e.g., '.pdf', '.txt')

        Returns:
            Path to stored file

        Raises:
            ValueError: If both file_path and content are provided or both are None
        """
        pass

    @abstractmethod
    def get_raw_file(self, file_id: str) -> Optional[Path]:
        """Get path to a raw file."""
        pass

    @abstractmethod
    def delete_raw_file(self, file_id: str) -> bool:
        """Delete a raw file."""
        pass

    # Knowledge Items Management
    @abstractmethod
    def store_knowledge(self, knowledge: KnowledgeItem) -> str:
        """Store a knowledge item."""
        pass

    @abstractmethod
    def get_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge item by ID."""
        pass

    @abstractmethod
    def delete_knowledge(self, knowledge_id: str) -> bool:
        """Delete a knowledge item."""
        pass

    # Embeddings Management
    @abstractmethod
    def store_embedding(
        self, knowledge_id: str, embedding: List[float], model: str
    ) -> str:
        """Store an embedding vector."""
        pass

    @abstractmethod
    def get_embedding(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding data for a knowledge item."""
        pass

    @abstractmethod
    def delete_embedding(self, knowledge_id: str) -> bool:
        """Delete an embedding."""
        pass

    # Extra Data Management
    @abstractmethod
    def store_extra(self, key: str, data: Any) -> str:
        """Store extra data (hashes, metadata, etc.)."""
        pass

    @abstractmethod
    def get_extra(self, key: str) -> Optional[Any]:
        """Get extra data by key."""
        pass

    @abstractmethod
    def delete_extra(self, key: str) -> bool:
        """Delete extra data."""
        pass

    # Specialized Knowledge Item Processing
    def process_knowledge(self, knowledge: KnowledgeItem) -> str:
        """Store knowledge item with specialized processing based on type.

        This method provides type-specific handling:
        - Paper: Download PDF if URL available and not already stored
        - Other types: Basic storage

        Args:
            knowledge: KnowledgeItem instance to store

        Returns:
            Knowledge ID after storage
        """
        knowledge_id = knowledge.get_primary_id()

        # Store the knowledge item first
        stored_id = self.store_knowledge(knowledge)

        # Type-specific processing
        if isinstance(knowledge, Paper):
            logger.info(
                f"Storage Processing paper {knowledge.get_primary_id()}"
            )
            self._handle_paper_files(knowledge)

        return stored_id

    def process_knowledges(self, knowledges: List[KnowledgeItem]) -> List[str]:
        """Process a list of knowledge items."""
        return [self.process_knowledge(knowledge) for knowledge in knowledges]

    def _handle_paper_files(self, paper: Paper) -> None:
        """Handle file operations for Paper objects.

        Args:
            paper: Paper instance to process
        """
        paper_id = paper.get_primary_id()

        # Check if PDF file already exists
        existing_pdf = self.get_raw_file(paper_id)
        if existing_pdf and existing_pdf.exists():
            return  # File already exists

        # Try to download PDF if URL is available
        if paper.pdf_url:
            try:
                content = self._download_file_content(paper.pdf_url)
                if content:
                    self.store_raw_file(
                        file_id=paper_id, content=content, file_extension=".pdf"
                    )
            except Exception as e:
                # Log error but don't fail the entire operation
                logger.error(
                    f"Failed to download PDF for {paper.get_primary_id()}: {e}"
                )

    def _download_file_content(self, url: str) -> Optional[bytes]:
        """Download file content from URL.

        Args:
            url: URL to download from

        Returns:
            File content as bytes or None if failed
        """
        try:
            import requests

            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception:
            return None

    # Utility Methods
    def get_all_knowledges(self) -> Iterator[KnowledgeItem]:
        """Get all knowledge items."""
        return iter(self.search_knowledges(limit=None))

    def knowledge_exists(self, knowledge_id: str) -> bool:
        """Check if a knowledge item exists."""
        return self.get_knowledge(knowledge_id) is not None

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information."""
        return {
            "type": self.__class__.__name__,
            "knowledge_count": len(list(self.get_all_knowledges())),
        }

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"
