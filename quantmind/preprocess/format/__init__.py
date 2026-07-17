"""Format layer — turns raw bytes into LLM-friendly markdown/text."""

from quantmind.preprocess.format.html import html_to_markdown
from quantmind.preprocess.format.pdf import (
    BoundingBox,
    ParsedChunk,
    ParsedDocument,
    ParsedDocumentHit,
    ParsedPage,
    PdfParseError,
    SentenceSplitterConfig,
    TextBlock,
    chunk_parsed_document,
    parse_pdf,
    pdf_to_markdown,
    retrieve_parsed_document,
)

__all__ = [
    "BoundingBox",
    "ParsedChunk",
    "ParsedDocument",
    "ParsedDocumentHit",
    "ParsedPage",
    "PdfParseError",
    "SentenceSplitterConfig",
    "TextBlock",
    "chunk_parsed_document",
    "html_to_markdown",
    "parse_pdf",
    "pdf_to_markdown",
    "retrieve_parsed_document",
]
