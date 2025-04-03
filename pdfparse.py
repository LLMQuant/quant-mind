from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
import subprocess
from autoscholar.base.parse_tool import ParseTool
from autoscholar.knowledge import Paper


class PDF2MarkdownTool(ParseTool):
    """Tool for converting PDFs to Markdown format using marker library."""

    def __init__(
        self,
        cleanup: bool = True,
        extract_images: bool = False,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the PDF to Markdown conversion tool.

        Args:
            cleanup: Whether to clean up the converted markdown content
            extract_images: Whether to extract images from the PDF
            output_dir: Output directory for converted files. If not specified, uses default directory
        """
        self.cleanup = cleanup
        self.extract_images = extract_images
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

    def parse(self, file_path: str) -> str:
        """Convert PDF to markdown using marker library.

        Args:
            file_path: Path to the PDF file

        Returns:
            String containing markdown representation of the PDF
        """
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        self.logger.info(f"Converting {file_path} to markdown")

        try:
            # Build marker command
            cmd = ["marker_single", str(pdf_path)]

            # Add optional parameters
            if self.output_dir:
                cmd.extend(["--output_dir", str(self.output_dir)])
            if self.extract_images:
                cmd.append("--extract_images")

            # Execute conversion command
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Check if successful
            if result.returncode != 0:
                raise Exception(f"Conversion failed: {result.stderr}")

            # Get output file path
            output_path = pdf_path.with_suffix(".md")
            if self.output_dir:
                output_path = Path(self.output_dir) / output_path.name

            # Read converted content
            markdown_content = output_path.read_text(encoding="utf-8")

            # Clean up content
            if self.cleanup:
                markdown_content = self._clean_markdown(markdown_content)

            self.logger.info("PDF conversion completed successfully")
            return markdown_content

        except Exception as e:
            self.logger.error(f"Error during conversion: {str(e)}")
            raise

    def get_format(self) -> str:
        return "markdown"

    def _clean_markdown(self, content: str) -> str:
        """Clean up the markdown content."""
        # Delete extra blank lines
        content = "\n".join(line for line in content.split("\n") if line.strip())

        # Ensure titles have blank lines before and after
        content = content.replace("\n#", "\n\n#")
        content = content.replace("\n##", "\n\n##")
        content = content.replace("\n###", "\n\n###")

        return content.strip()


class PDF2JSONTool(ParseTool):
    """Tool for converting PDFs to structured JSON format."""

    def __init__(self, include_metadata: bool = True, extract_references: bool = True):
        self.include_metadata = include_metadata
        self.extract_references = extract_references
        self.logger = logging.getLogger(__name__)

    def parse(self, file_path: str) -> Dict:
        """Convert PDF to JSON structure.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dict containing structured representation of the PDF
        """
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        self.logger.info(f"Converting {file_path} to JSON structure")

        # Extract document structure
        document_structure = self._extract_document_structure(str(pdf_path))

        # Extract metadata if requested
        if self.include_metadata:
            metadata = self._extract_metadata(str(pdf_path))
            document_structure["metadata"] = metadata

        # Extract references if requested
        if self.extract_references:
            references = self._extract_references(str(pdf_path))
            document_structure["references"] = references

        return document_structure

    def get_format(self) -> str:
        return "json"

    def _extract_document_structure(self, file_path: str) -> Dict:
        """Extract document structure from PDF."""
        # Actual implementation would go here
        return {
            "title": "Sample Paper Title",
            "sections": [
                {
                    "heading": "Introduction",
                    "level": 1,
                    "content": "This is the introduction content...",
                    "subsections": [],
                },
                # More sections...
            ],
        }

    def _extract_metadata(self, file_path: str) -> Dict:
        """Extract document metadata."""
        return {
            "authors": ["Author 1", "Author 2"],
            "year": 2023,
            "doi": "10.1234/5678",
            "journal": "Journal of Sample Science",
        }

    def _extract_references(self, file_path: str) -> List[Dict]:
        """Extract references from the document."""
        return [
            {
                "authors": ["Author A", "Author B"],
                "year": 2020,
                "title": "A referenced paper",
                "journal": "Journal of References",
                "doi": "10.5678/1234",
            },
            # More references...
        ]


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize parsing tools
    pdf2md = PDF2MarkdownTool(cleanup=True, extract_images=True)
    pdf2json = PDF2JSONTool(include_metadata=True, extract_references=True)

    # Create and parse a paper with markdown
    paper1 = Paper(path="example_paper.pdf")
    paper1.parse_pdf(pdf2md)
    markdown_content = paper1.get_content()
    paper1.save_parsed_content("example_paper.md")

    # Create and parse a paper with JSON
    paper2 = Paper(path="example_paper.pdf", title="Manual Title Override")
    paper2.parse_pdf(pdf2json)
    json_structure = paper2.get_content()
    paper2.save_parsed_content("example_paper.json")
