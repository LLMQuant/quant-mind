"""Unit tests for LlamaParser."""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

from quantmind.config.parsers import LlamaParserConfig, ParsingMode, ResultType
from quantmind.models.paper import Paper
from quantmind.parsers.llama_parser import LlamaParser


class TestLlamaParserConfig(unittest.TestCase):
    """Test cases for LlamaParserConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LlamaParserConfig()

        self.assertEqual(config.result_type, ResultType.MD)
        self.assertEqual(config.parsing_mode, ParsingMode.FAST)
        self.assertEqual(config.max_file_size_mb, 50)
        self.assertIsNone(config.api_key)
        self.assertIsNone(config.system_prompt)
        self.assertFalse(config.verbose)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = LlamaParserConfig(
            result_type=ResultType.TXT,
            parsing_mode=ParsingMode.PREMIUM,
            max_file_size_mb=25,
            system_prompt="Extract financial data",
            target_pages=[1, 2, 3, -1],
        )
        self.assertEqual(config.result_type, ResultType.TXT)
        self.assertEqual(config.parsing_mode, ParsingMode.PREMIUM)
        self.assertEqual(config.max_file_size_mb, 25)
        self.assertEqual(config.target_pages, [1, 2, 3, -1])

    def test_invalid_target_pages(self):
        """Test invalid target pages validation."""
        with self.assertRaises(ValueError):
            LlamaParserConfig(target_pages=[0])  # Page 0 is invalid

        with self.assertRaises(ValueError):
            LlamaParserConfig(target_pages=[-2])  # Invalid negative page

        with self.assertRaises(ValueError):
            LlamaParserConfig(target_pages=[])  # Empty list

    def test_invalid_language(self):
        """Test invalid language validation."""
        with self.assertRaises(ValueError):
            LlamaParserConfig(language="invalid")

    def test_get_llama_parse_config(self):
        """Test LlamaParse config generation."""
        config = LlamaParserConfig(
            result_type=ResultType.MD,
            parsing_mode=ParsingMode.BALANCED,
            system_prompt="Test prompt",
            num_workers=2,
            verbose=True,
        )

        llama_config = config.get_llama_parse_config()

        self.assertEqual(llama_config["result_type"], "markdown")
        self.assertFalse(llama_config["fast_mode"])
        self.assertFalse(llama_config["premium_mode"])
        self.assertEqual(llama_config["system_prompt"], "Test prompt")
        self.assertEqual(llama_config["num_workers"], 2)
        self.assertTrue(llama_config["verbose"])


class TestLlamaParser(unittest.TestCase):
    """Test cases for LlamaParser."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_api_key = "test_api_key"
        self.test_config = LlamaParserConfig(
            api_key=self.mock_api_key,
            result_type=ResultType.MD,
            parsing_mode=ParsingMode.FAST,
            max_file_size_mb=50,
        )

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_init_with_config(self, mock_llama_parse):
        """Test LlamaParser initialization with config."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)

        self.assertEqual(parser.api_key, self.mock_api_key)
        self.assertEqual(parser.llama_config.result_type, ResultType.MD)
        self.assertEqual(parser.llama_config.parsing_mode, ParsingMode.FAST)
        self.assertEqual(parser.llama_config.max_file_size_mb, 50)
        mock_llama_parse.assert_called_once()

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_init_with_dict_config(self, mock_llama_parse):
        """Test LlamaParser initialization with dict config."""
        mock_llama_parse.return_value = Mock()

        dict_config = {
            "api_key": self.mock_api_key,
            "result_type": "text",
            "parsing_mode": "premium",
        }

        parser = LlamaParser(dict_config)

        self.assertEqual(parser.api_key, self.mock_api_key)
        self.assertEqual(parser.llama_config.result_type, ResultType.TXT)
        self.assertEqual(parser.llama_config.parsing_mode, ParsingMode.PREMIUM)
        mock_llama_parse.assert_called_once()

    @patch.dict(os.environ, {"LLAMA_CLOUD_API_KEY": "env_api_key"})
    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_init_with_env_var(self, mock_llama_parse):
        """Test LlamaParser initialization with environment variable."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser()

        self.assertEqual(parser.api_key, "env_api_key")
        mock_llama_parse.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LlamaParser()
        self.assertIn("LLAMA_CLOUD_API_KEY is required", str(context.exception))

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    @patch("quantmind.parsers.llama_parser.requests.get")
    @patch("tempfile.NamedTemporaryFile")
    def test_parse_paper_with_pdf_url(
        self, mock_temp_file, mock_requests, mock_llama_parse
    ):
        """Test parsing paper with PDF URL."""
        # Mock temp file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.pdf"
        mock_temp_file.return_value.__enter__.return_value = mock_temp

        # Mock requests
        mock_response = Mock()
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.iter_content.return_value = [b"fake pdf content"]
        mock_requests.return_value = mock_response

        # Mock LlamaParse result
        mock_result = Mock()
        mock_result.get_markdown_documents.return_value = [
            Mock(
                text="This is a comprehensive extracted content from PDF URL that contains sufficient words and characters to pass validation checks. This represents a financial research paper's content that has been successfully processed by the LlamaParse API from a remote URL source."
            )
        ]

        mock_llama_instance = Mock()
        mock_llama_instance.parse.return_value = mock_result
        mock_llama_parse.return_value = mock_llama_instance

        parser = LlamaParser(self.test_config)
        paper = Paper(
            paper_id="test_paper",
            title="Test Paper",
            pdf_url="https://example.com/paper.pdf",
        )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch("os.path.exists", return_value=True),
            patch("os.unlink"),
        ):
            mock_stat.return_value.st_size = 1024 * 1024  # 1MB
            result = parser.parse_paper(paper)

        self.assertIsNotNone(result.content)
        self.assertIn(
            "comprehensive extracted content from PDF URL", result.content
        )
        self.assertEqual(
            result.meta_info["parser_info"]["parser"], "LlamaParser"
        )
        self.assertEqual(
            result.meta_info["parser_info"]["result_type"], "markdown"
        )
        mock_requests.assert_called_once()
        mock_llama_instance.parse.assert_called_once()

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_parse_paper_with_local_file_url(self, mock_llama_parse):
        """Test parsing paper with local file URL."""
        # Mock LlamaParse result
        mock_result = Mock()
        mock_result.get_markdown_documents.return_value = [
            Mock(
                text="This is a long extracted content from PDF that contains enough words and characters to pass the content validation checks in the LlamaParser implementation. It represents the full text content of a financial research paper that has been successfully parsed from the PDF document using the LlamaParse API."
            )
        ]

        mock_llama_instance = Mock()
        mock_llama_instance.parse.return_value = mock_result
        mock_llama_parse.return_value = mock_llama_instance

        parser = LlamaParser(self.test_config)
        paper = Paper(
            paper_id="test_paper",
            title="Test Paper",
            pdf_url="file:///path/to/test.pdf",
        )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch(
                "pathlib.Path.suffix", new_callable=PropertyMock
            ) as mock_suffix,
        ):
            mock_stat.return_value.st_size = 10 * 1024 * 1024  # 10MB
            mock_suffix.return_value = ".pdf"
            result = parser.parse_paper(paper)

        self.assertIsNotNone(result.content)
        self.assertIn("long extracted content from PDF", result.content)
        self.assertEqual(
            result.meta_info["parser_info"]["parser"], "LlamaParser"
        )
        self.assertEqual(
            result.meta_info["parser_info"]["result_type"], "markdown"
        )
        mock_llama_instance.parse.assert_called_once()

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_parse_paper_file_too_large(self, mock_llama_parse):
        """Test parsing paper with file too large."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)
        paper = Paper(
            paper_id="test_paper",
            title="Test Paper",
            pdf_url="file:///path/to/large.pdf",
        )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100MB
            result = parser.parse_paper(paper)

        # Should return original paper without content due to size limit
        self.assertIsNone(result.content)
        self.assertNotIn("parser_info", result.meta_info)

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_parse_paper_no_pdf(self, mock_llama_parse):
        """Test parsing paper without PDF."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)
        paper = Paper(paper_id="test_paper", title="Test Paper")

        result = parser.parse_paper(paper)

        # Should return original paper unchanged
        self.assertIsNone(result.content)
        self.assertNotIn("parser_info", result.meta_info)

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_parse_paper_parsing_error(self, mock_llama_parse):
        """Test parsing paper with parsing error."""
        mock_llama_instance = Mock()
        mock_llama_instance.parse.side_effect = Exception("Parsing failed")
        mock_llama_parse.return_value = mock_llama_instance

        parser = LlamaParser(self.test_config)
        paper = Paper(
            paper_id="test_paper",
            title="Test Paper",
            pdf_url="https://example.com/paper.pdf",
        )

        result = parser.parse_paper(paper)

        # Should return original paper without content due to error
        self.assertIsNone(result.content)
        self.assertNotIn("parser_info", result.meta_info)

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_parse_content_pdf(self, mock_llama_parse):
        """Test parsing PDF content."""
        # Mock LlamaParse result
        mock_result = Mock()
        mock_result.get_markdown_documents.return_value = [
            Mock(
                text="This is a detailed parsed PDF content that contains enough words and characters to pass the validation requirements in the LlamaParser implementation for testing purposes."
            )
        ]

        mock_llama_instance = Mock()
        mock_llama_instance.parse.return_value = mock_result
        mock_llama_parse.return_value = mock_llama_instance

        parser = LlamaParser(self.test_config)

        with patch("pathlib.Path.exists", return_value=True):
            result = parser.parse_content("test.pdf", "pdf")

        self.assertIn("detailed parsed PDF content", result)
        mock_llama_instance.parse.assert_called_once_with("test.pdf")

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_parse_content_non_pdf(self, mock_llama_parse):
        """Test parsing non-PDF content."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)
        result = parser.parse_content("test content", "text")

        # Should return empty string for non-PDF
        self.assertEqual(result, "")

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_extract_from_file_success(self, mock_llama_parse):
        """Test extracting content from file successfully."""
        # Mock LlamaParse result
        mock_result = Mock()
        mock_result.get_markdown_documents.return_value = [
            Mock(
                text="This is comprehensive file content extracted from PDF that contains enough text and words to satisfy the content validation requirements for testing the file extraction functionality."
            )
        ]

        mock_llama_instance = Mock()
        mock_llama_instance.parse.return_value = mock_result
        mock_llama_parse.return_value = mock_llama_instance

        parser = LlamaParser(self.test_config)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch(
                "pathlib.Path.suffix", new_callable=PropertyMock
            ) as mock_suffix,
        ):
            mock_stat.return_value.st_size = 10 * 1024 * 1024  # 10MB
            mock_suffix.return_value = ".pdf"
            result = parser.extract_from_file("test.pdf")

        self.assertIn("comprehensive file content extracted", result)

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_extract_from_file_not_exists(self, mock_llama_parse):
        """Test extracting from non-existent file."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)

        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                parser.extract_from_file("nonexistent.pdf")

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_extract_from_file_non_pdf(self, mock_llama_parse):
        """Test extracting from non-PDF file."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)

        with patch("pathlib.Path.exists", return_value=True):
            with self.assertRaises(NotImplementedError) as context:
                parser.extract_from_file("test.txt")
            self.assertIn("only supports PDF files", str(context.exception))

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_extract_from_file_too_large(self, mock_llama_parse):
        """Test extracting from oversized file."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100MB
            with self.assertRaises(ValueError) as context:
                parser.extract_from_file("large.pdf")
            self.assertIn("PDF file too large", str(context.exception))

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    @patch("quantmind.parsers.llama_parser.requests.get")
    @patch("tempfile.NamedTemporaryFile")
    def test_extract_from_url_success(
        self, mock_temp_file, mock_requests, mock_llama_parse
    ):
        """Test extracting content from URL successfully."""
        # Mock temp file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.pdf"
        mock_temp_file.return_value.__enter__.return_value = mock_temp

        # Mock requests
        mock_response = Mock()
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.iter_content.return_value = [b"fake pdf content"]
        mock_requests.return_value = mock_response

        # Mock LlamaParse result
        mock_result = Mock()
        mock_result.get_markdown_documents.return_value = [
            Mock(
                text="This is detailed URL content extracted from PDF document that has been successfully processed and contains sufficient text length to pass validation checks for testing purposes."
            )
        ]

        mock_llama_instance = Mock()
        mock_llama_instance.parse.return_value = mock_result
        mock_llama_parse.return_value = mock_llama_instance

        parser = LlamaParser(self.test_config)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch(
                "pathlib.Path.suffix", new_callable=PropertyMock
            ) as mock_suffix,
            patch("os.path.exists", return_value=True),
            patch("os.unlink"),
        ):
            mock_stat.return_value.st_size = 1024 * 1024  # 1MB
            mock_suffix.return_value = ".pdf"
            result = parser.extract_from_url("https://example.com/paper.pdf")

        self.assertIn("detailed URL content extracted", result)

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_extract_from_url_non_pdf(self, mock_llama_parse):
        """Test extracting from non-PDF URL."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)

        with self.assertRaises(ValueError) as context:
            parser.extract_from_url("https://example.com/page.html")
        self.assertIn("only supports PDF URLs", str(context.exception))

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_extract_content_from_result_various_attributes(
        self, mock_llama_parse
    ):
        """Test extracting content from JobResult with new API."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)

        # Test with markdown documents
        result_with_markdown = Mock()
        result_with_markdown.get_markdown_documents.return_value = [
            Mock(text="Markdown content page 1"),
            Mock(text="Markdown content page 2"),
        ]
        self.assertEqual(
            parser._extract_content_from_result(result_with_markdown),
            "Markdown content page 1\n\nMarkdown content page 2",
        )

        # Test with text documents (create a new parser with text result type)
        text_config = LlamaParserConfig(
            api_key=self.mock_api_key,
            result_type=ResultType.TXT,
            parsing_mode=ParsingMode.FAST,
        )
        text_parser = LlamaParser(text_config)

        result_with_text = Mock()
        result_with_text.get_text_documents.return_value = [
            Mock(text="Text content page 1"),
            Mock(text="Text content page 2"),
        ]
        self.assertEqual(
            text_parser._extract_content_from_result(result_with_text),
            "Text content page 1\n\nText content page 2",
        )

        # Test fallback with JSON result type (should fall back to pages)
        json_config = LlamaParserConfig(
            api_key=self.mock_api_key,
            result_type=ResultType.JSON,  # This will trigger else branch
            parsing_mode=ParsingMode.FAST,
        )
        json_parser = LlamaParser(json_config)

        # Create mock pages with proper attributes
        page1 = Mock()
        page1.md = "Page 1 markdown"
        page1.text = "Page 1 text"

        page2 = Mock()
        page2.md = "Page 2 markdown"
        page2.text = "Page 2 text"

        result_with_pages = Mock()
        result_with_pages.pages = [page1, page2]

        content = json_parser._extract_content_from_result(result_with_pages)
        self.assertIn("Page 1 markdown", content)
        self.assertIn("Page 2 markdown", content)

        # Test empty pages fallback
        result_empty = Mock()
        result_empty.pages = []
        result_empty.__str__ = Mock(return_value="Fallback string result")

        content = json_parser._extract_content_from_result(result_empty)
        self.assertEqual(content, "Fallback string result")

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_get_parser_info(self, mock_llama_parse):
        """Test getting parser information."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)
        info = parser.get_parser_info()

        self.assertEqual(info["name"], "llama")
        self.assertEqual(info["type"], "LlamaParser")
        self.assertEqual(info["result_type"], "markdown")
        self.assertEqual(info["max_file_size_mb"], 50)
        self.assertTrue(info["supports_urls"])
        self.assertTrue(info["supports_local_files"])

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_validate_content_quality(self, mock_llama_parse):
        """Test content validation."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)

        # Valid content
        good_content = "This is a valid paper content with sufficient length and meaningful text that should pass validation checks. It contains enough words and characters to meet the minimum requirements for content validation in the LlamaParser implementation."
        self.assertTrue(parser.validate_content(good_content))

        # Empty content
        self.assertFalse(parser.validate_content(""))
        self.assertFalse(parser.validate_content(None))

    @patch("quantmind.parsers.llama_parser.LlamaParse")
    def test_clean_text_functionality(self, mock_llama_parse):
        """Test text cleaning functionality."""
        mock_llama_parse.return_value = Mock()

        parser = LlamaParser(self.test_config)

        # Test cleaning various text issues
        dirty_text = (
            "Text with  multiple   spaces\n\n\nand\tmultiple\nlines\x0c"
        )
        cleaned = parser.clean_text(dirty_text)

        self.assertNotIn("  ", cleaned)  # No double spaces
        self.assertNotIn("\x0c", cleaned)  # No form feed
        self.assertEqual(
            cleaned.strip(), cleaned
        )  # No leading/trailing whitespace


class TestParsingModeEnum(unittest.TestCase):
    """Test cases for ParsingMode enum."""

    def test_parsing_mode_values(self):
        """Test ParsingMode enum values."""
        self.assertEqual(ParsingMode.FAST.value, "fast")
        self.assertEqual(ParsingMode.BALANCED.value, "balanced")
        self.assertEqual(ParsingMode.PREMIUM.value, "premium")


class TestResultTypeEnum(unittest.TestCase):
    """Test cases for ResultType enum."""

    def test_result_type_values(self):
        """Test ResultType enum values."""
        self.assertEqual(ResultType.TXT.value, "text")
        self.assertEqual(ResultType.MD.value, "markdown")
        self.assertEqual(ResultType.JSON.value, "json")


if __name__ == "__main__":
    unittest.main()
