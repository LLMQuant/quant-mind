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
        """Extract document structure from PDF.

        TODO: Implement actual logic for extracting document structure from PDF.
        This should include:
        1. Parsing the PDF file to identify sections and subsections
        2. Extracting headings and their hierarchy levels
        3. Extracting content for each section
        4. Building a structured dictionary with title, sections, and their content

        Returns:
            Dict: A dictionary containing the document structure with the following format:
            {
                "title": str,  # Document title
                "sections": [  # List of sections
                    {
                        "heading": str,  # Section heading
                        "level": int,    # Heading level (1 for main sections, 2 for subsections, etc.)
                        "content": str,  # Section content
                        "subsections": []  # Nested subsections
                    }
                ]
            }
        """
        pass

    def _extract_metadata(self, file_path: str) -> Dict:
        """Extract document metadata.

        TODO: Implement actual logic for extracting metadata from PDF.
        This should include:
        1. Extracting author information
        2. Extracting publication year
        3. Extracting DOI (Digital Object Identifier)
        4. Extracting journal/conference information
        5. Extracting any other relevant metadata

        Returns:
            Dict: A dictionary containing the document metadata with the following format:
            {
                "authors": List[str],  # List of author names
                "year": int,           # Publication year
                "doi": str,            # Digital Object Identifier
                "journal": str,        # Journal or conference name
                "keywords": List[str], # Optional: List of keywords
                "abstract": str        # Optional: Abstract text
            }
        """
        pass

    def _extract_references(self, file_path: str) -> List[Dict]:
        """Extract references from the document.

        TODO: Implement actual logic for extracting references from PDF.
        This should include:
        1. Identifying reference section
        2. Parsing individual references
        3. Extracting reference details (authors, year, title, etc.)
        4. Validating and formatting reference data

        Returns:
            List[Dict]: A list of dictionaries, each containing a reference with the following format:
            [
                {
                    "authors": List[str],  # List of author names
                    "year": int,           # Publication year
                    "title": str,          # Paper title
                    "journal": str,        # Journal or conference name
                    "doi": str,            # Digital Object Identifier
                    "volume": str,         # Optional: Journal volume
                    "issue": str,          # Optional: Journal issue
                    "pages": str           # Optional: Page numbers
                }
            ]
        """
        pass


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
