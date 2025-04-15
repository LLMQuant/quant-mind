import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from autoscholar.knowledge.paper import Paper
from autoscholar.parser.pdf_parser import (
    MarkdownConverter,
    MarkerSingleConverter,
    PDF2MarkdownTool,
)


class TestMarkdownConverter(unittest.TestCase):
    """Test MarkdownConverter abstract base class interface."""

    def test_abstract_class(self):
        """Test that MarkdownConverter is correctly defined as an abstract class."""
        with self.assertRaises(TypeError):
            MarkdownConverter()


class MockMarkdownConverter(MarkdownConverter):
    """Mock implementation of MarkdownConverter for testing."""

    def convert(self, pdf_path, **options):
        """Return mock Markdown content."""
        return "# Test Title\n\nTest content\n\n## Test Section\n\nMore content"


class TestMarkerSingleConverter(unittest.TestCase):
    """Test MarkerSingleConverter class."""

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    def test_convert_success(self, mock_read_text, mock_exists, mock_run):
        """Test successful PDF conversion."""
        # Set mock return values
        mock_exists.return_value = True
        mock_read_text.return_value = "# Test Document\n\nTest content"

        # Mock successful command execution
        process_mock = MagicMock()
        process_mock.returncode = 0
        mock_run.return_value = process_mock

        # Execute conversion
        converter = MarkerSingleConverter()
        result = converter.convert(Path("test.pdf"))

        # Verify result
        self.assertEqual(result, "# Test Document\n\nTest content")
        mock_run.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_convert_file_not_found(self, mock_exists):
        """Test file not found case."""
        mock_exists.return_value = False

        converter = MarkerSingleConverter()
        with self.assertRaises(FileNotFoundError):
            converter.convert(Path("nonexistent.pdf"))

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_convert_command_failed(self, mock_exists, mock_run):
        """Test case when command execution fails."""
        mock_exists.return_value = True

        # Mock command execution failure
        process_mock = MagicMock()
        process_mock.returncode = 1
        process_mock.stderr = "Error: Conversion failed"
        mock_run.return_value = process_mock

        converter = MarkerSingleConverter()
        with self.assertRaises(Exception):
            converter.convert(Path("test.pdf"))


class TestPDF2MarkdownTool(unittest.TestCase):
    """Test PDF2MarkdownTool class."""

    def setUp(self):
        """Set up test environment."""
        # Use mock converter instead of actual marker_single tool
        self.mock_converter = MockMarkdownConverter()
        self.pdf_tool = PDF2MarkdownTool(converter=self.mock_converter)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        # Clean up files in temporary directory
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_parse(self):
        """Test parse method."""
        result = self.pdf_tool.parse("test.pdf")
        self.assertIsInstance(result, str)
        self.assertIn("Test Title", result)
        self.assertIn("Test Section", result)

    def test_get_format(self):
        """Test get_format method."""
        from autoscholar.parser.parse_tool import STRUCTURED_TYPES

        self.assertEqual(self.pdf_tool.get_format(), STRUCTURED_TYPES.MARKDOWN)

    def test_clean_markdown(self):
        """Test _clean_markdown method."""
        markdown = "# Title\n\n\n\nContent\n\n\n## Section\nMore content"
        cleaned = self.pdf_tool._clean_markdown(markdown)

        # Verify extra blank lines have been removed
        self.assertNotIn("\n\n\n", cleaned)

        # Verify titles have blank lines before and after
        self.assertIn("\n\n## Section", cleaned)

    def test_parse_to_paper(self):
        """Test parse_to_paper method."""
        # Mock parsed Paper object
        # Set the cleanup as False to avoid removing the blank lines
        pdf_tool = PDF2MarkdownTool(
            converter=self.mock_converter, cleanup=False
        )
        paper = pdf_tool.parse_to_paper("test.pdf", title="Test Paper")
        pdf_tool.cleanup = True

        # Verify Paper object attributes
        self.assertEqual(paper.title, "Test Paper")
        self.assertEqual(paper.full_text, self.mock_converter.convert(None))
        self.assertIn("markdown_content", paper.meta_info)
        self.assertTrue(paper.pdf_url.startswith("file://"))

    def test_save_paper_content(self):
        """Test save_paper_content method."""
        # Create a Paper object with content
        paper = Paper(
            title="Test Save", meta_info={"markdown_content": "Test content"}
        )

        # Save to temporary file
        output_file = os.path.join(self.temp_dir, "test_paper.json")
        result_path = self.pdf_tool.save_paper_content(paper, output_file)

        # Verify file has been created
        self.assertTrue(os.path.exists(output_file))

        # Verify returned path is correct
        self.assertEqual(result_path, output_file)

        # Verify file content
        with open(output_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
            self.assertEqual(loaded_data["title"], "Test Save")
            self.assertIn("markdown_content", loaded_data["meta_info"])

    def test_auto_generate_output_path(self):
        """Test auto generate output path when no output path is provided."""
        # Create a Paper object with content
        paper = Paper(
            title="Auto Generate",
            meta_info={"markdown_content": "Test content"},
        )

        # Execute test in temporary directory
        current_dir = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            result_path = self.pdf_tool.save_paper_content(paper)

            # Verify file has been created, filename based on paper title
            expected_filename = "Auto_Generate.json"
            self.assertTrue(os.path.exists(expected_filename))
            self.assertEqual(result_path, expected_filename)
        finally:
            os.chdir(current_dir)

    def test_error_when_no_meta_info(self):
        """Test error handling when Paper object has no meta info."""
        paper = Paper(title="No Meta Info")  # No meta_info set

        with self.assertRaises(ValueError):
            self.pdf_tool.save_paper_content(paper)


if __name__ == "__main__":
    unittest.main()
