"""Tests for Paper model."""

import pytest
from datetime import datetime
from quantmind.models.paper import Paper


class TestPaper:
    """Test cases for the Paper model."""

    def test_paper_creation(self):
        """Test basic paper creation."""
        paper = Paper(title="Test Paper", abstract="This is a test abstract")

        assert paper.title == "Test Paper"
        assert paper.abstract == "This is a test abstract"
        assert isinstance(paper.id, str)
        assert len(paper.categories) == 0
        assert len(paper.tags) == 0

    def test_paper_with_metadata(self):
        """Test paper creation with metadata."""
        published_date = datetime(2023, 1, 15)

        paper = Paper(
            title="Advanced ML Paper",
            abstract="This paper discusses advanced machine learning techniques",
            authors=["John Doe", "Jane Smith"],
            published_date=published_date,
            categories=["Machine Learning"],
            tags=["deep learning", "neural networks"],
            arxiv_id="2301.12345",
        )

        assert paper.title == "Advanced ML Paper"
        assert len(paper.authors) == 2
        assert paper.published_date == published_date
        assert "Machine Learning" in paper.categories
        assert "deep learning" in paper.tags
        assert paper.arxiv_id == "2301.12345"

    def test_add_category(self):
        """Test adding categories."""
        paper = Paper(title="Test", abstract="Test abstract")

        paper.add_category("Finance")
        paper.add_category("Machine Learning")
        paper.add_category("Finance")  # Duplicate

        assert len(paper.categories) == 2
        assert "Finance" in paper.categories
        assert "Machine Learning" in paper.categories

    def test_add_tag(self):
        """Test adding tags."""
        paper = Paper(title="Test", abstract="Test abstract")

        paper.add_tag("lstm")
        paper.add_tag("trading")
        paper.add_tag("lstm")  # Duplicate

        assert len(paper.tags) == 2
        assert "lstm" in paper.tags
        assert "trading" in paper.tags

    def test_get_text_for_embedding(self):
        """Test text extraction for embedding."""
        paper = Paper(
            title="ML in Finance",
            abstract="Machine learning applications in financial markets",
        )

        text = paper.get_text_for_embedding()
        expected = "ML in Finance\n\nMachine learning applications in financial markets"
        assert text == expected

    def test_set_embedding(self):
        """Test setting embedding."""
        paper = Paper(title="Test", abstract="Test abstract")
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        paper.set_embedding(embedding, "text-embedding-ada-002")

        assert paper.embedding == embedding
        assert paper.embedding_model == "text-embedding-ada-002"

    def test_has_full_text(self):
        """Test full text availability check."""
        paper1 = Paper(title="Test", abstract="Test abstract")
        paper2 = Paper(
            title="Test",
            abstract="Test abstract",
            full_text="Full paper content",
        )
        paper3 = Paper(title="Test", abstract="Test abstract", full_text="   ")

        assert not paper1.has_full_text()
        assert paper2.has_full_text()
        assert not paper3.has_full_text()

    def test_get_primary_id(self):
        """Test primary ID extraction."""
        paper1 = Paper(
            title="Test", abstract="Test abstract", arxiv_id="2301.12345"
        )
        paper2 = Paper(
            title="Test", abstract="Test abstract", paper_id="custom_id"
        )
        paper3 = Paper(title="Test", abstract="Test abstract")

        assert paper1.get_primary_id() == "2301.12345"
        assert paper2.get_primary_id() == "custom_id"
        assert paper3.get_primary_id() == paper3.id

    def test_from_dict(self):
        """Test creating paper from dictionary."""
        data = {
            "title": "Test Paper",
            "abstract": "Test abstract",
            "authors": ["Author 1", "Author 2"],
            "categories": ["AI", "ML"],
            "arxiv_id": "2301.12345",
            "published_date": "2023-01-15T00:00:00",
        }

        paper = Paper.from_dict(data)

        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert len(paper.categories) == 2
        assert paper.arxiv_id == "2301.12345"
        assert isinstance(paper.published_date, datetime)

    def test_dict_conversion(self):
        """Test paper to dictionary conversion."""
        paper = Paper(
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author 1"],
            categories=["AI"],
            tags=["test"],
        )

        data = paper.dict()

        assert data["title"] == "Test Paper"
        assert data["abstract"] == "Test abstract"
        assert data["authors"] == ["Author 1"]
        assert data["categories"] == ["AI"]
        assert data["tags"] == ["test"]

    def test_authors_parsing(self):
        """Test author parsing from various formats."""
        # String format
        paper1 = Paper(
            title="Test", abstract="Test", authors="John Doe, Jane Smith"
        )
        assert len(paper1.authors) == 2
        assert "John Doe" in paper1.authors

        # List format
        paper2 = Paper(
            title="Test", abstract="Test", authors=["John Doe", "Jane Smith"]
        )
        assert len(paper2.authors) == 2

        # Empty
        paper3 = Paper(title="Test", abstract="Test", authors=None)
        assert len(paper3.authors) == 0

    def test_validation(self):
        """Test paper validation."""
        # Valid paper
        paper1 = Paper(
            title="Valid Title",
            abstract="Valid abstract with sufficient length",
        )
        assert len(paper1.title) >= 1
        assert len(paper1.abstract) >= 1

        # Test minimum requirements are enforced by Pydantic
        with pytest.raises(ValueError):
            Paper(title="", abstract="Valid abstract")

        # Test empty abstract is allowed
        Paper(title="Valid title", abstract="")

    def test_string_representations(self):
        """Test string representations."""
        paper = Paper(
            title="Test Paper", abstract="Test abstract", arxiv_id="2301.12345"
        )

        str_repr = str(paper)
        repr_repr = repr(paper)

        assert "2301.12345" in str_repr
        assert "Test Paper" in str_repr
        assert "Test Paper" in repr_repr
