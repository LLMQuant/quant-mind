"""Format layer — turns raw bytes into LLM-friendly markdown/text."""

from quantmind.preprocess.format.html import html_to_markdown
from quantmind.preprocess.format.pdf import PdfParseError, pdf_to_markdown

__all__ = [
    "PdfParseError",
    "html_to_markdown",
    "pdf_to_markdown",
]
