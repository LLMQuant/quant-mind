"""Tests for ArXiv source functionality."""

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from quantmind.config.sources import ArxivSourceConfig
from quantmind.models.paper import Paper
from quantmind.sources.arxiv_source import ArxivSource


class TestArxivSourceConfig(unittest.TestCase):
    """Test cases for ArxivSourceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ArxivSourceConfig()

        assert config.max_results == 100
        assert config.sort_by == "submittedDate"
        assert config.sort_order == "descending"
        assert config.download_pdfs is False
        assert config.requests_per_second == 1.0
        assert config.min_abstract_length == 50
        assert config.proxies is None

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ArxivSourceConfig(
            sort_by="relevance",
            sort_order="ascending",
            max_results=50,
            proxies={
                "http": "http://localhost:8080",
                "https": "http://localhost:8080",
            },
        )
        assert config.sort_by == "relevance"
        assert config.proxies == {
            "http": "http://localhost:8080",
            "https": "http://localhost:8080",
        }

        # Invalid sort_by
        with pytest.raises(ValueError):
            ArxivSourceConfig(sort_by="invalid")

        # Invalid sort_order
        with pytest.raises(ValueError):
            ArxivSourceConfig(sort_order="invalid")

    def test_download_dir_creation(self):
        """Test download directory creation."""
        with TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "nonexistent"
            config = ArxivSourceConfig(download_dir=test_dir)

            assert config.download_dir.exists()
            assert config.download_dir.is_dir()

    def test_arxiv_criterion_conversion(self):
        """Test arXiv criterion conversion."""
        config = ArxivSourceConfig(sort_by="relevance", sort_order="ascending")

        # Test sort criterion
        criterion = config.get_arxiv_sort_criterion()
        assert hasattr(criterion, "value")  # Should be arxiv.SortCriterion

        # Test sort order
        order = config.get_arxiv_sort_order()
        assert hasattr(order, "value")  # Should be arxiv.SortOrder


class TestArxivSource(unittest.TestCase):
    """Test cases for ArxivSource."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ArxivSourceConfig(
            max_results=5, download_pdfs=False, min_abstract_length=10
        )
        self.source = ArxivSource(config=self.config)

    def test_initialization(self):
        """Test source initialization."""
        # Test with config object
        source = ArxivSource(config=self.config)
        assert isinstance(source.config, ArxivSourceConfig)

        # Test with dict config
        dict_config = {"max_results": 10}
        source = ArxivSource(config=dict_config)
        assert source.config.max_results == 10

        # Test with no config
        source = ArxivSource()
        assert isinstance(source.config, ArxivSourceConfig)

    @patch("quantmind.sources.arxiv_source.arxiv.Client")
    def test_rate_limiting(self, mock_client):
        """Test rate limiting functionality."""
        source = ArxivSource(config=ArxivSourceConfig(requests_per_second=2.0))

        # Mock time to test rate limiting
        with patch("time.time", side_effect=[1, 1.2, 1.5, 1.7, 3.0]):
            with patch("time.sleep") as mock_sleep:
                source._rate_limit()  # First call, no sleep
                source._rate_limit()  # Second call, should sleep

                mock_sleep.assert_called_once()
                # Should sleep for ~0.2 seconds (0.5 - 0.3)
                sleep_time = mock_sleep.call_args[0][0]
                assert 0.1 < sleep_time < 0.3

    def test_should_include_paper(self):
        """Test paper filtering logic."""
        paper1 = Paper(
            title="Test Paper",
            abstract="This is a long enough abstract for testing purposes",
            categories=["cs.AI"],
        )

        paper2 = Paper(
            title="Short Abstract Paper",
            abstract="Short",  # Too short
            categories=["cs.AI"],
        )

        paper3 = Paper(
            title="Wrong Category Paper",
            abstract="This is a long enough abstract",
            categories=["bio.GN"],  # Not in include list
        )

        # Test default filtering
        assert self.source._should_include_paper(paper1) is True
        assert self.source._should_include_paper(paper2) is False

        # Test with category filtering
        config_with_filters = ArxivSourceConfig(
            include_categories=["cs.AI"], min_abstract_length=10
        )
        source_filtered = ArxivSource(config=config_with_filters)

        assert source_filtered._should_include_paper(paper1) is True
        assert source_filtered._should_include_paper(paper3) is False

    def test_clean_arxiv_id(self):
        """Test arXiv ID cleaning."""
        test_cases = [
            ("arXiv:2301.12345", "2301.12345"),
            ("arxiv:2301.12345", "2301.12345"),
            ("http://arxiv.org/abs/2301.12345", "2301.12345"),
            ("https://arxiv.org/abs/2301.12345", "2301.12345"),
            ("2301.12345", "2301.12345"),
            ("  2301.12345  ", "2301.12345"),
        ]

        for input_id, expected in test_cases:
            result = self.source._clean_arxiv_id(input_id)
            assert result == expected, f"Failed for input: {input_id}"

    @patch("quantmind.sources.arxiv_source.requests.get")
    def test_download_pdf_success(self, mock_get):
        """Test successful PDF download."""
        # Mock successful response
        mock_response = Mock()
        mock_response.content = b"fake pdf content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        paper = Paper(
            title="Test Paper",
            abstract="Test abstract",
            arxiv_id="2301.12345",
            pdf_url="https://arxiv.org/pdf/2301.12345.pdf",
        )

        with TemporaryDirectory() as temp_dir:
            download_dir = Path(temp_dir)
            result = self.source.download_pdf(paper, download_dir)

            assert result is not None
            assert result.exists()
            assert result.name.startswith("2301.12345")
            assert result.suffix == ".pdf"

            # Check content was written
            assert result.read_bytes() == b"fake pdf content"

    @patch("quantmind.sources.arxiv_source.requests.get")
    def test_download_pdf_failure(self, mock_get):
        """Test PDF download failure."""
        # Mock failed response
        mock_get.side_effect = Exception("Network error")

        paper = Paper(
            title="Test Paper",
            abstract="Test abstract",
            pdf_url="https://arxiv.org/pdf/2301.12345.pdf",
        )

        with TemporaryDirectory() as temp_dir:
            result = self.source.download_pdf(paper, Path(temp_dir))
            assert result is None

    def test_download_pdf_no_url(self):
        """Test PDF download with no URL."""
        paper = Paper(
            title="Test Paper", abstract="Test abstract", pdf_url=None
        )

        with TemporaryDirectory() as temp_dir:
            result = self.source.download_pdf(paper, Path(temp_dir))
            assert result is None

    @patch("quantmind.sources.arxiv_source.arxiv.Client")
    def test_validate_config(self, mock_client_class):
        """Test configuration validation."""
        # Mock successful client
        mock_client = Mock()
        mock_client.results.return_value = iter([])
        mock_client_class.return_value = mock_client

        source = ArxivSource()
        assert source.validate_config() is True

        # Mock failed client
        mock_client.results.side_effect = Exception("Connection failed")
        assert source.validate_config() is False

    def test_convert_arxiv_result(self):
        """Test conversion from arXiv result to Paper."""
        # Mock arXiv result
        mock_result = Mock()
        mock_result.entry_id = "http://arxiv.org/abs/2301.12345v1"
        mock_result.title = "Test Paper Title"
        mock_result.summary = "This is a test abstract"
        mock_result.authors = [
            Mock(__str__=lambda x: "John Doe"),
            Mock(__str__=lambda x: "Jane Smith"),
        ]
        mock_result.published = datetime(2023, 1, 15)
        mock_result.categories = ["cs.AI", "stat.ML"]
        mock_result.pdf_url = "https://arxiv.org/pdf/2301.12345.pdf"
        mock_result.doi = "10.1000/test"
        mock_result.journal_ref = "Test Journal 2023"
        mock_result.primary_category = "cs.AI"
        mock_result.comment = "Test comment"
        mock_result.links = [Mock(href="https://example.com")]

        paper = self.source._convert_arxiv_result(mock_result)

        assert isinstance(paper, Paper)
        assert paper.arxiv_id == "2301.12345v1"
        assert paper.title == "Test Paper Title"
        assert paper.abstract == "This is a test abstract"
        assert paper.authors == ["John Doe", "Jane Smith"]
        assert paper.published_date == datetime(2023, 1, 15)
        assert paper.categories == ["cs.AI", "stat.ML"]
        assert paper.pdf_url == "https://arxiv.org/pdf/2301.12345.pdf"
        assert paper.source == "arxiv"
        assert paper.meta_info["doi"] == "10.1000/test"
