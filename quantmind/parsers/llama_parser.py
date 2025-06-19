"""LlamaParser implementation for AI-powered PDF parsing."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from llama_cloud_services import LlamaParse
from llama_cloud_services.parse.types import JobResult

from quantmind.config.parsers import LlamaParserConfig
from quantmind.models.paper import Paper
from quantmind.utils.env import get_llama_cloud_api_key
from quantmind.utils.logger import get_logger

from .base import BaseParser


class LlamaParser(BaseParser):
    """LlamaParser implementation using LlamaParse API.

    This parser uses LlamaParse for AI-powered PDF parsing with advanced
    document understanding capabilities. It only supports local file parsing,
    so URLs are automatically downloaded to temporary files.

    For more information, see: https://docs.cloud.llamaindex.ai/llamaparse/
    """

    def __init__(
        self, config: Optional[Union[LlamaParserConfig, Dict[str, Any]]] = None
    ):
        """Initialize LlamaParser with configuration.

        Args:
            config: Parser configuration (LlamaParserConfig or dict)

        Raises:
            ValueError: If LLAMA_CLOUD_API_KEY is not set
        """
        # Handle config conversion
        if isinstance(config, LlamaParserConfig):
            llama_config = config
        elif isinstance(config, dict):
            llama_config = LlamaParserConfig(**config)
        else:
            llama_config = LlamaParserConfig()

        # Initialize base parser with the config
        super().__init__(llama_config)
        self.logger = get_logger(__name__)

        # Store typed config for easier access
        self.llama_config = llama_config

        # Extract API key using modern environment management
        self.api_key = self.llama_config.api_key or get_llama_cloud_api_key(
            required=False
        )
        if not self.api_key:
            raise ValueError(
                "LLAMA_CLOUD_API_KEY is required for LlamaParser. "
                "Set it as environment variable, in .env file, or in config."
            )

        # Initialize LlamaParse client with simplified config
        try:
            self.llama_parse = LlamaParse(
                api_key=self.api_key,
                result_type=self._get_result_type_value(),
                num_workers=self.llama_config.num_workers or 1,
                verbose=self.llama_config.verbose,
                language=self.llama_config.language,
            )
            self.logger.info("LlamaParser initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaParser: {e}")
            raise

    def _get_result_type_value(self) -> str:
        """Get result type as string value."""
        if isinstance(self.llama_config.result_type, str):
            return self.llama_config.result_type
        else:
            return self.llama_config.result_type.value

    def parse_paper(self, paper: Paper) -> Paper:
        """Parse and enrich paper content.

        Args:
            paper: Paper object to parse

        Returns:
            Paper object with enriched content
        """
        if not paper.pdf_url:
            self.logger.warning(
                f"No PDF available for paper {paper.get_primary_id()}"
            )
            return paper

        try:
            self.logger.info(
                f"Parsing paper {paper.get_primary_id()} with LlamaParse"
            )

            # Determine if it's a local file or URL
            if paper.pdf_url.startswith(("http://", "https://")):
                content = self.extract_from_url(paper.pdf_url)
            elif paper.pdf_url.startswith("file://"):
                local_path = paper.pdf_url[7:]  # Remove file:// prefix
                content = self.extract_from_file(local_path)
            else:
                # Assume it's a local file path
                content = self.extract_from_file(paper.pdf_url)

            # Validate and store content
            if content and self.validate_content(content):
                paper.content = self.clean_text(content)
                paper.meta_info["parser_info"] = {
                    "parser": "LlamaParser",
                    "result_type": self._get_result_type_value(),
                    "content_length": len(content),
                }
                self.logger.info(
                    f"Successfully parsed paper {paper.get_primary_id()} "
                    f"({len(content)} characters)"
                )
            else:
                self.logger.warning(
                    f"Parsed content failed validation for paper "
                    f"{paper.get_primary_id()}"
                )

        except Exception as e:
            self.logger.error(
                f"Error parsing paper {paper.get_primary_id()} with LlamaParse: {e}"
            )

        return paper

    def parse_content(self, content: str, content_type: str = "text") -> str:
        """Parse raw content into structured text.

        Args:
            content: File path to parse (must be local file)
            content_type: Type of content (pdf, etc.)

        Returns:
            Parsed and cleaned text content
        """
        if content_type != "pdf":
            self.logger.warning(
                f"LlamaParser only supports PDF content, got: {content_type}"
            )
            return ""

        try:
            # LlamaParse only accepts local file paths
            local_path = Path(content)
            if not local_path.exists():
                raise FileNotFoundError(f"File not found: {local_path}")

            self.logger.debug(f"Parsing local file: {local_path}")
            result = self.llama_parse.parse(str(local_path))
            self.logger.debug(f"Result: {result}")

            # Extract content from result using new API
            content_text = self._extract_content_from_result(result)

            if self.validate_content(content_text):
                return self.clean_text(content_text)
            else:
                self.logger.warning("Parsed content failed validation")
                return ""

        except Exception as e:
            self.logger.error(f"Error parsing content with LlamaParse: {e}")
            return ""

    def extract_from_file(self, file_path: str) -> str:
        """Extract content from a local PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise NotImplementedError("LlamaParser now only supports PDF files")

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.llama_config.max_file_size_mb:
            raise ValueError(
                f"PDF file too large ({file_size_mb:.1f}MB > "
                f"{self.llama_config.max_file_size_mb}MB)"
            )

        return self.parse_content(str(file_path), "pdf")

    def extract_from_url(self, url: str) -> str:
        """Extract content from a PDF URL by downloading it first.

        Args:
            url: URL to PDF file

        Returns:
            Extracted text content
        """
        if not url.lower().endswith(".pdf") and "pdf" not in url.lower():
            raise ValueError("LlamaParser only supports PDF URLs")

        temp_file = None
        try:
            # Download PDF to temporary file
            self.logger.debug(f"Downloading PDF from URL: {url}")
            response = requests.get(
                url, timeout=self.llama_config.timeout, stream=True
            )
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                raise ValueError(
                    f"URL does not point to a PDF file. Content-Type: {content_type}"
                )

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as temp_file:
                temp_path = temp_file.name

                # Download with size limit
                downloaded_size = 0
                max_size = self.llama_config.max_file_size_mb * 1024 * 1024

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > max_size:
                            raise ValueError(
                                f"Downloaded file too large (>{self.llama_config.max_file_size_mb}MB)"
                            )
                        temp_file.write(chunk)

            self.logger.debug(
                f"Downloaded {downloaded_size} bytes to {temp_path}"
            )

            # Parse the downloaded file
            return self.extract_from_file(temp_path)

        except Exception as e:
            self.logger.error(
                f"Error downloading/parsing PDF from URL {url}: {e}"
            )
            raise
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    self.logger.debug(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to clean up temporary file {temp_path}: {e}"
                    )

    def _extract_content_from_result(self, result: JobResult) -> str:
        """Extract content from LlamaParse JobResult using new API.

        Args:
            result: JobResult from LlamaParse

        Returns:
            Extracted content as string
        """
        try:
            result_type = self._get_result_type_value()
            self.logger.debug(f"Result type: {result_type}")

            if result_type == "markdown":
                # Get markdown documents and join them
                docs = result.get_markdown_documents()
                return "\n\n".join(doc.text for doc in docs)
            elif result_type == "text":
                # Get text documents and join them
                docs = result.get_text_documents()
                return "\n\n".join(doc.text for doc in docs)
            else:
                # Fallback: try to access raw text from pages
                content_parts = []
                for page in result.pages:
                    if hasattr(page, "md") and page.md:
                        content_parts.append(page.md)
                    elif hasattr(page, "text") and page.text:
                        content_parts.append(page.text)

                if content_parts:
                    return "\n\n".join(content_parts)
                else:
                    return str(result)

        except Exception as e:
            self.logger.error(f"Failed to extract content from result: {e}")
            # Fallback to string representation
            return str(result)

    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about this parser.

        Returns:
            Dictionary with parser metadata
        """
        info = super().get_parser_info()
        info.update(
            {
                "result_type": self._get_result_type_value(),
                "parsing_mode": self.llama_config.parsing_mode,
                "max_file_size_mb": self.llama_config.max_file_size_mb,
                "num_workers": self.llama_config.num_workers,
                "language": self.llama_config.language,
                "supports_urls": True,  # Now supports URLs via download
                "supports_local_files": True,
            }
        )
        return info
