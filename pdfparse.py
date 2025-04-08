from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
import subprocess
import json
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
        # 临时实现，返回一个示例结构
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
        # 临时实现，返回一个示例结构
        return {
            "authors": ["Author 1", "Author 2"],
            "year": 2023,
            "doi": "10.1234/5678",
            "journal": "Journal of Sample Science",
        }

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
        # 临时实现，返回一个示例结构
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


class PDFParser:
    """Adapter class to integrate PDF parsing with the Paper class from the main repository."""

    def __init__(self, parse_tool: ParseTool):
        """
        Initialize the PDF parser.

        Args:
            parse_tool: An instance of a class implementing the ParseTool interface
        """
        self.parse_tool = parse_tool
        self.logger = logging.getLogger(__name__)

    def parse_pdf_to_paper(self, pdf_path: str, title: Optional[str] = None) -> Paper:
        """
        Parse a PDF file and create a Paper object.

        Args:
            pdf_path: Path to the PDF file
            title: Optional title for the paper

        Returns:
            Paper object with parsed content
        """
        try:
            # Parse the PDF
            content = self.parse_tool.parse(pdf_path)

            # Create a Paper object
            paper = Paper(title=title or "")

            # Store the parsed content in meta_info
            if self.parse_tool.get_format() == "markdown":
                paper.meta_info["markdown_content"] = content
                paper.full_text = content
            elif self.parse_tool.get_format() == "json":
                # If it's JSON, extract title and other metadata if available
                if isinstance(content, dict):
                    if "title" in content and not paper.title:
                        paper.title = content["title"]

                    # Store the entire JSON structure in meta_info
                    paper.meta_info["json_content"] = content

                    # Extract abstract if available
                    if "metadata" in content and "abstract" in content["metadata"]:
                        paper.abstract = content["metadata"]["abstract"]

            # Set the PDF URL
            paper.pdf_url = f"file://{Path(pdf_path).absolute()}"

            return paper

        except Exception as e:
            self.logger.error(f"Error parsing PDF: {str(e)}")
            raise

    def save_paper_content(self, paper: Paper, output_path: Optional[str] = None) -> str:
        """
        Save the paper content to a file.

        Args:
            paper: Paper object with parsed content
            output_path: Custom path to save the file. If not provided, will use a default name.

        Returns:
            Path to the saved file
        """
        if not paper.meta_info:
            raise ValueError("Paper has no content to save.")

        if output_path is None:
            # Generate output path based on paper title or ID
            base_name = paper.title or paper.id
            # Replace spaces and special characters
            base_name = "".join(c if c.isalnum() else "_" for c in base_name)
            output_path = f"{base_name}.json"

        try:
            output_path = Path(output_path)

            # Save as JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(paper.to_dict(), f, ensure_ascii=False, indent=2)

            self.logger.info(f"Saved paper content to {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Error saving paper content: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize parsing tools
    pdf2md = PDF2MarkdownTool(cleanup=True, extract_images=True)
    pdf2json = PDF2JSONTool(include_metadata=True, extract_references=True)

    # Create PDF parser
    pdf_parser = PDFParser(pdf2md)

    # Parse PDF to Paper
    paper = pdf_parser.parse_pdf_to_paper("example_paper.pdf")
    print(f"Parsed paper: {paper.title}")

    # Save paper content
    output_path = pdf_parser.save_paper_content(paper)
    print(f"Saved paper content to: {output_path}")
