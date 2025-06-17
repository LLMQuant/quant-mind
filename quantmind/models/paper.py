"""Paper model for QuantMind knowledge representation."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class Paper(BaseModel):
    """Research paper entity with structured metadata and validation.

    Core knowledge unit in the QuantMind system, representing a research paper
    with comprehensive metadata, content, and processing information.
    """

    # Core identifiers
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None

    # Core content
    title: str = Field(..., min_length=1)
    abstract: str = Field(..., min_length=1)
    full_text: Optional[str] = None

    # Metadata
    authors: List[str] = Field(default_factory=list)
    published_date: Optional[datetime] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    # URLs and sources
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    code_url: Optional[str] = None

    # Processing metadata
    source: Optional[str] = None  # e.g., "arxiv", "pubmed"
    extraction_method: Optional[str] = None  # e.g., "api", "pdf_parse"
    processed_at: datetime = Field(default_factory=datetime.utcnow)

    # Vector representation
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None

    # Flexible metadata storage
    meta_info: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    @field_validator("categories", "tags", mode="before")
    def ensure_list(cls, v):
        """Ensure categories and tags are always lists."""
        if isinstance(v, str):
            return [v]
        return v or []

    @field_validator("authors", mode="before")
    def parse_authors(cls, v):
        """Parse authors from various formats."""
        if isinstance(v, str):
            # Handle comma-separated authors
            return [author.strip() for author in v.split(",")]
        return v or []

    def get_text_for_embedding(self) -> str:
        """Get concatenated text for embedding generation.

        Returns:
            Combined title and abstract text
        """
        return f"{self.title}\n\n{self.abstract}"

    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present.

        Args:
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)

    def add_category(self, category: str) -> None:
        """Add a category if not already present.

        Args:
            category: Category to add
        """
        if category not in self.categories:
            self.categories.append(category)

    def set_embedding(self, embedding: List[float], model: str = None) -> None:
        """Set the paper's embedding vector.

        Args:
            embedding: Vector representation
            model: Name of the embedding model used
        """
        self.embedding = embedding
        if model:
            self.embedding_model = model

    def has_full_text(self) -> bool:
        """Check if paper has full text content.

        Returns:
            True if full text is available
        """
        return bool(self.full_text and len(self.full_text.strip()) > 0)

    def get_primary_id(self) -> str:
        """Get the primary identifier for the paper.

        Returns:
            ArXiv ID if available, otherwise paper_id, otherwise id
        """
        return self.arxiv_id or self.paper_id or self.id

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Paper":
        """Create Paper from dictionary with flexible field mapping.

        Args:
            data: Dictionary with paper data

        Returns:
            Paper instance
        """
        # Handle datetime fields
        if "published_date" in data and isinstance(data["published_date"], str):
            try:
                data["published_date"] = datetime.fromisoformat(
                    data["published_date"]
                )
            except ValueError:
                data["published_date"] = None

        if "processed_at" in data and isinstance(data["processed_at"], str):
            try:
                data["processed_at"] = datetime.fromisoformat(
                    data["processed_at"]
                )
            except ValueError:
                data["processed_at"] = datetime.utcnow()

        return cls(**data)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "Paper":
        """Load paper from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Paper instance
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_from_files(
        cls, file_paths: List[Union[str, Path]]
    ) -> List["Paper"]:
        """Load multiple papers from JSON files.

        Args:
            file_paths: List of file paths

        Returns:
            List of Paper instances
        """
        return [cls.load_from_file(path) for path in file_paths]

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save paper to JSON file.

        Args:
            file_path: Output file path
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                self.model_dump(), f, ensure_ascii=False, indent=2, default=str
            )

    def __str__(self) -> str:
        """String representation."""
        return f"Paper({self.get_primary_id()}): {self.title[:50]}..."

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Paper(id='{self.id}', title='{self.title[:30]}...', source='{self.source}')"
