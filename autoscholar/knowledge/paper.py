from typing import Dict, Optional, Union
from pathlib import Path
import logging
from autoscholar.base.parse_tool import ParseTool


class Paper:
    """Class representing an academic paper with PDF parsing capabilities."""

    def __init__(self, path: Optional[str] = None, title: Optional[str] = None):
        """Initialize a Paper instance.

        Args:
            path: Path to the PDF file
            title: Optional title of the paper
        """
        self.path = path
        self.title = title
        self.content = None
        self.content_format = None
        self.metadata = {}
        self.logger = logging.getLogger(__name__)

    def parse_pdf(self, parse_tool: ParseTool) -> None:
        """Parse the PDF using the provided parsing tool.

        Args:
            parse_tool: An instance of a class implementing the ParseTool interface
        """
        if not self.path:
            raise ValueError("Paper path not set. Set paper.path before parsing.")

        try:
            self.logger.info(f"Parsing PDF using {parse_tool.__class__.__name__}")
            self.content = parse_tool.parse(self.path)
            self.content_format = parse_tool.get_format()

            # Extract title if not already set
            if not self.title and self.content_format == "json":
                self.title = self.content.get("title", None)

            self.logger.info(f"Successfully parsed PDF to {self.content_format} format")

        except Exception as e:
            self.logger.error(f"Error parsing PDF: {str(e)}")
            raise

    def get_content(self) -> Union[str, Dict]:
        """Get the parsed content.

        Returns:
            The parsed content in the format specified by the parse_tool used
        """
        if self.content is None:
            raise ValueError("Paper has not been parsed yet. Call parse_pdf first.")
        return self.content

    def save_parsed_content(self, output_path: Optional[str] = None) -> str:
        """Save the parsed content to a file.

        Args:
            output_path: Custom path to save the file. If not provided, will use the PDF name.

        Returns:
            Path to the saved file
        """
        if self.content is None:
            raise ValueError("No content to save. Parse the PDF first.")

        if output_path is None:
            # Generate output path based on original PDF path
            pdf_path = Path(self.path)
            ext = ".md" if self.content_format == "markdown" else ".json"
            output_path = pdf_path.with_suffix(ext)

        try:
            output_path = Path(output_path)
            if self.content_format == "json":
                import json

                output_path.write_text(
                    json.dumps(self.content, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                output_path.write_text(self.content, encoding="utf-8")

            self.logger.info(f"Saved parsed content to {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Error saving content: {str(e)}")
            raise
