import json
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..knowledge import Paper
from ..utils.logger import setup_logger
from .parse_tool import STRUCTURED_TYPES, ParseTool


class MarkdownConverter(ABC):
    """Abstract base class for PDF to Markdown converters."""

    @abstractmethod
    def convert(self, pdf_path: Path, **options) -> str:
        """Convert PDF to markdown.

        Args:
            pdf_path: Path to the PDF file
            **options: Additional options for the conversion

        Returns:
            String containing markdown representation of the PDF
        """
        pass


class MarkerSingleConverter(MarkdownConverter):
    """Converter that uses marker_single tool."""

    def convert(
        self,
        pdf_path: Path,
        extract_images: bool = False,
        output_dir: Optional[str] = None,
        **options,
    ) -> str:
        """Convert PDF to markdown using marker library.

        Args:
            pdf_path: Path to the PDF file
            extract_images: Whether to extract images from the PDF
            output_dir: Output directory for converted files
            **options: Additional options for marker_single

        Returns:
            String containing markdown representation of the PDF
        """
        logger = setup_logger(__name__)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Converting {pdf_path} to markdown using marker_single")

        try:
            # Build marker command
            cmd = ["marker_single", str(pdf_path)]

            # Add optional parameters
            if output_dir:
                cmd.extend(["--output_dir", str(output_dir)])
            if extract_images:
                cmd.append("--extract_images")

            # Execute conversion command
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Check if successful
            if result.returncode != 0:
                raise Exception(f"Conversion failed: {result.stderr}")

            # Get output file path
            output_path = pdf_path.with_suffix(".md")
            if output_dir:
                output_path = Path(output_dir) / output_path.name

            # Read converted content
            markdown_content = output_path.read_text(encoding="utf-8")

            logger.info("PDF conversion completed successfully")
            return markdown_content

        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            raise


class PDF2MarkdownTool(ParseTool):
    """Tool for converting PDFs to Markdown format."""

    def __init__(
        self,
        converter: Optional[MarkdownConverter] = None,
        cleanup: bool = True,
        extract_images: bool = False,
        output_dir: Optional[str] = None,
    ):
        """Initialize the PDF to Markdown conversion tool.

        Args:
            converter: MarkdownConverter implementation to use
            cleanup: Whether to clean up the converted markdown content
            extract_images: Whether to extract images from the PDF
            output_dir: Output directory for converted files
        """
        self.converter = converter or MarkerSingleConverter()
        self.cleanup = cleanup
        self.extract_images = extract_images
        self.output_dir = output_dir
        self.logger = setup_logger(__name__)

    def parse(
        self, source: Union[str, Path, bytes], **kwargs: Any
    ) -> Union[str, Dict, Any]:
        """Convert PDF to markdown.

        Args:
            source: Path to the PDF file or Path object
            **kwargs: Additional arguments

        Returns:
            String containing markdown representation of the PDF
        """
        pdf_path = Path(source) if isinstance(source, str) else source

        # Use the converter to convert PDF to markdown
        options = {
            "extract_images": self.extract_images,
            "output_dir": self.output_dir,
            **kwargs,
        }

        markdown_content = self.converter.convert(pdf_path, **options)

        # Clean up content if needed
        if self.cleanup:
            markdown_content = self._clean_markdown(markdown_content)

        return markdown_content

    def get_format(self) -> STRUCTURED_TYPES:
        return STRUCTURED_TYPES.MARKDOWN

    def _clean_markdown(self, content: str) -> str:
        """Clean up the markdown content."""
        # Delete extra blank lines
        content = "\n".join(
            line for line in content.split("\n") if line.strip()
        )

        # Ensure titles have blank lines before and after
        # Use regex to avoid multiple replacements
        content = re.sub(r"\n(#+)", r"\n\n\1", content)

        return content.strip()

    def parse_to_paper(
        self, source: Union[str, Path], title: Optional[str] = None
    ) -> Paper:
        """Parse a PDF file and create a Paper object.

        Args:
            source: Path to the PDF file
            title: Optional title for the paper

        Returns:
            Paper object with parsed content
        """
        try:
            # Parse the PDF
            content = self.parse(source)

            # Create a Paper object
            paper = Paper(title=title or "")

            # Store the parsed content
            paper.meta_info["markdown_content"] = content
            paper.full_text = content

            # Set the PDF URL
            pdf_path = Path(source) if isinstance(source, str) else source
            paper.pdf_url = f"file://{pdf_path.absolute()}"

            return paper

        except Exception as e:
            self.logger.error(f"Error parsing PDF: {str(e)}")
            raise

    def save_paper_content(
        self, paper: Paper, output_path: Optional[str] = None
    ) -> str:
        """Save the paper content to a file.

        If the output_path is not provided,
        the file will be saved in the current working directory
        with the name of the paper title or ID. e.g. "Hello_World.json"

        Args:
            paper: Paper object with parsed content
            output_path: Custom path to save the file

        Returns:
            Path to the saved file
        """
        if not paper.meta_info:
            raise ValueError("Paper has no content to save.")

        if output_path is None:
            # Generate output path based on paper title or ID
            base_name = paper.title or paper.id
            # Replace spaces and special characters
            # e.g. "Hello World" -> "Hello_World"
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
    # Initialize parsing tools with default MarkerSingleConverter
    pdf2md = PDF2MarkdownTool(
        converter=MarkerSingleConverter(), cleanup=True, extract_images=True
    )

    # Use PDF2MarkdownTool to parse and create a Paper object
    paper = pdf2md.parse_to_paper("example_paper.pdf")
    print(f"Parsed paper: {paper.title}")

    # Save paper content
    output_path = pdf2md.save_paper_content(paper)
    print(f"Saved paper content to: {output_path}")
