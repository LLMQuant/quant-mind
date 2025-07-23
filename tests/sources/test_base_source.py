"""Tests for base source functionality."""

import unittest
from typing import List, Optional

from quantmind.models.content import KnowledgeItem
from quantmind.sources.base import BaseSource


class MockContent(KnowledgeItem):
    """Mock content for testing."""

    def get_text_for_embedding(self) -> str:
        return f"{self.title}\n{self.abstract or ''}"


class MockSource(BaseSource[MockContent]):
    """Mock source implementation for testing."""

    def __init__(self, config=None, mock_data=None):
        super().__init__(config)
        self.mock_data = mock_data or []

    def search(self, query: str, max_results: int = 10) -> List[MockContent]:
        # Simple mock search - return first max_results items
        return self.mock_data[:max_results]

    def get_by_id(self, content_id: str) -> Optional[MockContent]:
        # Find by source_id or id
        for item in self.mock_data:
            if item.source_id == content_id or item.id == content_id:
                return item
        return None

    def get_by_timeframe(
        self, days: int = 7, categories: Optional[List[str]] = None, **kwargs
    ) -> List[MockContent]:
        # Mock timeframe - just return all data
        result = self.mock_data
        if categories:
            result = [
                item
                for item in result
                if any(cat in item.categories for cat in categories)
            ]
        return result


class TestBaseSource(unittest.TestCase):
    """Test cases for BaseSource."""

    def setUp(self):
        """Set up test data."""
        self.mock_content = [
            MockContent(
                source_id="1",
                title="Test Paper 1",
                abstract="This is a test abstract",
                categories=["cs.AI"],
                source="test",
            ),
            MockContent(
                source_id="2",
                title="Test Paper 2",
                abstract="Another test abstract",
                categories=["stat.ML"],
                source="test",
            ),
            MockContent(
                source_id="3",
                title="Test Paper 3",
                abstract="Third test abstract",
                categories=["cs.AI", "stat.ML"],
                source="test",
            ),
        ]
        self.source = MockSource(mock_data=self.mock_content)

    def test_initialization(self):
        """Test source initialization."""
        config = {"test_key": "test_value"}
        source = MockSource(config=config)

        assert source.config == config
        assert source.name == "mock"

    def test_search(self):
        """Test search functionality."""
        results = self.source.search("test query", max_results=2)

        assert len(results) == 2
        assert all(isinstance(item, MockContent) for item in results)

    def test_get_by_id(self):
        """Test get by ID functionality."""
        # Test with source_id
        result = self.source.get_by_id("1")
        assert result is not None
        assert result.source_id == "1"

        # Test with non-existent ID
        result = self.source.get_by_id("nonexistent")
        assert result is None

    def test_get_batch(self):
        """Test batch retrieval."""
        ids = ["1", "2", "nonexistent"]
        results = self.source.get_batch(ids)

        assert len(results) == 2  # Only found items
        assert results[0].source_id == "1"
        assert results[1].source_id == "2"

    def test_validate_config(self):
        """Test config validation."""
        assert self.source.validate_config() is True

    def test_get_source_info(self):
        """Test source info retrieval."""
        info = self.source.get_source_info()

        assert "name" in info
        assert "type" in info
        assert info["name"] == "mock"
        assert info["type"] == "MockSource"

    def test_string_representation(self):
        """Test string representations."""
        str_repr = str(self.source)
        assert "MockSource" in str_repr
        assert "mock" in str_repr
