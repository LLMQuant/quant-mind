"""Unit tests for PDFParser."""

import unittest
from unittest.mock import MagicMock, Mock, patch

from quantmind.config.parsers import PDFParserConfig
from quantmind.models.paper import Paper
from quantmind.parsers.pdf_parser import PDFParser


class TestPDFParserConfig(unittest.TestCase):
    """Test cases for PDFParserConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PDFParserConfig()

        self.assertEqual(config.method, "pymupdf")
        self.assertTrue(config.download_pdfs)
        self.assertEqual(config.max_file_size_mb, 50)
        self.assertFalse(config.extract_images)
        self.assertTrue(config.extract_tables)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = PDFParserConfig(
            method="pdfplumber",
            download_pdfs=False,
            max_file_size_mb=25,
            extract_images=True,
            extract_tables=False,
        )
        self.assertEqual(config.method, "pdfplumber")
        self.assertFalse(config.download_pdfs)
        self.assertEqual(config.max_file_size_mb, 25)
        self.assertTrue(config.extract_images)
        self.assertFalse(config.extract_tables)

    def test_invalid_method(self):
        """Test invalid method validation."""
        with self.assertRaises(ValueError):
            PDFParserConfig(method="invalid_method")


class TestPDFParser(unittest.TestCase):
    """Test cases for PDFParser."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = PDFParserConfig(
            method="pymupdf",
            download_pdfs=True,
            max_file_size_mb=50,
        )

    @patch("quantmind.parsers.pdf_parser.PYMUPDF_AVAILABLE", True)
    def test_init_with_config(self):
        """Test PDFParser initialization with config."""
        parser = PDFParser(self.test_config)

        self.assertEqual(parser.method, "pymupdf")
        self.assertTrue(parser.download_pdfs)
        self.assertEqual(parser.max_file_size, 50 * 1024 * 1024)

    @patch("quantmind.parsers.pdf_parser.PYMUPDF_AVAILABLE", True)
    def test_init_with_dict_config(self):
        """Test PDFParser initialization with dict config."""
        dict_config = {
            "method": "pdfplumber",
            "download_pdfs": False,
            "max_file_size_mb": 25,
        }

        parser = PDFParser(dict_config)

        self.assertEqual(parser.method, "pdfplumber")
        self.assertFalse(parser.download_pdfs)
        self.assertEqual(parser.max_file_size, 25 * 1024 * 1024)

    @patch("quantmind.parsers.pdf_parser.PYMUPDF_AVAILABLE", True)
    def test_init_with_no_config(self):
        """Test PDFParser initialization with no config."""
        parser = PDFParser()

        # Should use defaults from PDFParserConfig
        self.assertEqual(parser.method, "pymupdf")
        self.assertTrue(parser.download_pdfs)
        self.assertEqual(parser.max_file_size, 50 * 1024 * 1024)

    @patch("quantmind.parsers.pdf_parser.PYMUPDF_AVAILABLE", True)
    def test_parse_paper_no_pdf(self):
        """Test parsing paper without PDF."""
        parser = PDFParser(self.test_config)
        paper = Paper(paper_id="test_paper", title="Test Paper")

        result = parser.parse_paper(paper)

        # Should return original paper unchanged
        self.assertIsNone(result.content)

    @patch("quantmind.parsers.pdf_parser.PYMUPDF_AVAILABLE", True)
    def test_get_parser_info(self):
        """Test getting parser information."""
        parser = PDFParser(self.test_config)
        info = parser.get_parser_info()

        self.assertEqual(info["name"], "pdf")
        self.assertEqual(info["type"], "PDFParser")
        self.assertIn("config", info)


if __name__ == "__main__":
    unittest.main()
