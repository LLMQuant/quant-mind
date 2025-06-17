"""PDF parser for extracting content from PDF files."""

import os
import tempfile
import requests
from typing import Dict, Optional, Any
from pathlib import Path

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models

    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False

from quantmind.models.paper import Paper
from quantmind.parsers.base import BaseParser
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class PDFParser(BaseParser):
    """PDF parser for extracting text content from PDF files.

    Supports multiple extraction methods:
    - PyMuPDF (simple text extraction)
    - Marker (AI-powered PDF to Markdown conversion)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PDF parser.

        Args:
            config: Configuration dictionary with:
                - method: 'pymupdf' or 'marker' (default: 'pymupdf')
                - download_pdfs: Whether to download PDFs from URLs (default: True)
                - cache_dir: Directory to cache downloaded PDFs (default: temp)
                - max_file_size: Maximum PDF file size in MB (default: 50)
        """
        super().__init__(config)

        self.method = self.config.get("method", "pymupdf")
        self.download_pdfs = self.config.get("download_pdfs", True)
        self.cache_dir = self.config.get("cache_dir", tempfile.gettempdir())
        self.max_file_size = (
            self.config.get("max_file_size", 50) * 1024 * 1024
        )  # Convert to bytes

        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Initialize marker models if using marker method
        self.marker_models = None
        if self.method == "marker" and MARKER_AVAILABLE:
            try:
                self.marker_models = load_all_models()
                logger.info("Marker models loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Marker models: {e}")
                self.method = "pymupdf"

        logger.info(f"PDFParser initialized with method: {self.method}")

    def parse_paper(self, paper: Paper) -> Paper:
        """Parse PDF content for a paper.

        Args:
            paper: Paper object to parse

        Returns:
            Paper object with extracted full text
        """
        if not paper.pdf_url:
            logger.debug(f"No PDF URL for paper {paper.get_primary_id()}")
            return paper

        try:
            # Download or get PDF content
            pdf_path = self._get_pdf_file(paper.pdf_url, paper.get_primary_id())
            if not pdf_path:
                return paper

            # Extract content
            if self.method == "marker" and self.marker_models:
                content = self._extract_with_marker(pdf_path)
            else:
                content = self._extract_with_pymupdf(pdf_path)

            if content and self.validate_content(content):
                paper.full_text = self.clean_text(content)
                paper.meta_info["pdf_parsed"] = True
                paper.meta_info["parser_method"] = self.method
                logger.debug(
                    f"Successfully parsed PDF for paper {paper.get_primary_id()}"
                )
            else:
                logger.warning(
                    f"Failed content validation for paper {paper.get_primary_id()}"
                )

        except Exception as e:
            logger.error(
                f"Error parsing PDF for paper {paper.get_primary_id()}: {e}"
            )
            paper.meta_info["pdf_parse_error"] = str(e)

        return paper

    def parse_content(
        self, content: str, content_type: str = "pdf_path"
    ) -> str:
        """Parse PDF content.

        Args:
            content: PDF file path or binary content
            content_type: Type of content ('pdf_path' or 'pdf_binary')

        Returns:
            Extracted text content
        """
        if content_type == "pdf_path":
            return self.extract_from_file(content)
        elif content_type == "pdf_binary":
            # Save binary content to temp file and process
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as tmp_file:
                tmp_file.write(
                    content.encode() if isinstance(content, str) else content
                )
                tmp_path = tmp_file.name

            try:
                return self.extract_from_file(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    def extract_from_file(self, file_path: str) -> str:
        """Extract content from a PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if self.method == "marker" and self.marker_models:
            return self._extract_with_marker(file_path)
        else:
            return self._extract_with_pymupdf(file_path)

    def extract_from_url(self, url: str) -> str:
        """Extract content from a PDF URL.

        Args:
            url: URL to PDF file

        Returns:
            Extracted text content
        """
        pdf_path = self._download_pdf(url)
        if pdf_path:
            try:
                return self.extract_from_file(pdf_path)
            finally:
                # Clean up downloaded file
                try:
                    os.unlink(pdf_path)
                except:
                    pass
        return ""

    def _get_pdf_file(self, pdf_url: str, paper_id: str) -> Optional[str]:
        """Get PDF file path (download if necessary).

        Args:
            pdf_url: URL to PDF file
            paper_id: Paper identifier for caching

        Returns:
            Path to PDF file, None if failed
        """
        if not self.download_pdfs:
            return None

        # Check cache first
        cache_filename = f"{paper_id}.pdf"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        if os.path.exists(cache_path):
            logger.debug(f"Using cached PDF: {cache_path}")
            return cache_path

        # Download PDF
        return self._download_pdf(pdf_url, cache_path)

    def _download_pdf(
        self, url: str, output_path: Optional[str] = None
    ) -> Optional[str]:
        """Download PDF from URL.

        Args:
            url: PDF URL
            output_path: Optional output path (default: temp file)

        Returns:
            Path to downloaded file, None if failed
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Check file size
            if len(response.content) > self.max_file_size:
                logger.warning(
                    f"PDF file too large: {len(response.content)} bytes"
                )
                return None

            # Determine output path
            if not output_path:
                output_path = tempfile.mktemp(suffix=".pdf")

            # Save file
            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.debug(f"Downloaded PDF: {url} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to download PDF from {url}: {e}")
            return None

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF not available. Install with: pip install pymupdf"
            )

        try:
            doc = fitz.open(pdf_path)
            text_parts = []

            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)

            doc.close()
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return ""

    def _extract_with_marker(self, pdf_path: str) -> str:
        """Extract text using Marker (AI-powered PDF to Markdown).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted markdown text
        """
        if not MARKER_AVAILABLE or not self.marker_models:
            raise ImportError("Marker not available or models not loaded")

        try:
            # Convert PDF to markdown
            full_text, images, out_meta = convert_single_pdf(
                pdf_path,
                self.marker_models,
                max_pages=None,
                langs=None,
                batch_multiplier=1,
            )

            logger.debug(
                f"Marker extraction completed: {len(full_text)} characters"
            )
            return full_text

        except Exception as e:
            logger.error(f"Marker extraction failed: {e}")
            return ""

    def validate_content(self, content: str) -> bool:
        """Validate PDF extraction quality.

        Args:
            content: Extracted content

        Returns:
            True if content meets quality standards
        """
        if not super().validate_content(content):
            return False

        # PDF-specific validation
        # Check for garbled text (too many non-ASCII characters)
        ascii_chars = sum(1 for c in content if ord(c) < 128)
        if len(content) > 0 and ascii_chars / len(content) < 0.7:
            return False

        # Check for extraction artifacts
        artifacts = ["�", "□", "▢", "cid:", "Obj"]
        artifact_count = sum(content.count(artifact) for artifact in artifacts)
        if artifact_count > len(content) / 100:  # More than 1% artifacts
            return False

        return True
