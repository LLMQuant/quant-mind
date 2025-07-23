"""Generic content model for QuantMind knowledge representation."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class BaseContent(BaseModel, ABC):
    """Abstract base class for all content types in QuantMind.

    This serves as the foundation for different knowledge entities
    like papers, articles, reports, etc.
    """

    # Core identifiers
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: Optional[str] = None  # ID from original source

    # Core content
    title: str = Field(..., min_length=1)
    abstract: Optional[str] = None
    content: Optional[str] = None  # Full content text

    # Metadata
    authors: List[str] = Field(default_factory=list)
    published_date: Optional[datetime] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    # Source information
    url: Optional[str] = None
    source: Optional[str] = None  # e.g., "arxiv", "pubmed", "news"
    extraction_method: Optional[str] = None  # e.g., "api", "scraping"
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Flexible metadata storage
    meta_info: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    @abstractmethod
    def get_text_for_embedding(self) -> str:
        """Get text content for embedding generation.

        Returns:
            String representation of content for vectorization
        """
        pass

    def get_primary_id(self) -> str:
        """Get the primary identifier for the content.

        Returns:
            Source ID if available, otherwise internal ID
        """
        return self.source_id or self.id

    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)

    def add_category(self, category: str) -> None:
        """Add a category if not already present."""
        if category not in self.categories:
            self.categories.append(category)


class KnowledgeItem(BaseContent):
    """Generic knowledge item implementation.

    Can represent various types of content like papers, articles,
    reports, news items, etc.
    """

    # Content-specific fields
    content_type: str = Field(default="generic")  # paper, article, report, etc.
    language: Optional[str] = None

    # Additional URLs
    pdf_url: Optional[str] = None
    code_url: Optional[str] = None

    # Vector representation
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None

    def get_text_for_embedding(self) -> str:
        """Get concatenated text for embedding generation."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.abstract:
            parts.append(self.abstract)
        if self.content:
            parts.append(self.content[:1000])  # Limit content length

        return "\n\n".join(parts)

    def has_full_content(self) -> bool:
        """Check if item has full content."""
        return bool(self.content and len(self.content.strip()) > 0)

    def set_embedding(self, embedding: List[float], model: str = None) -> None:
        """Set the content's embedding vector."""
        self.embedding = embedding
        if model:
            self.embedding_model = model
