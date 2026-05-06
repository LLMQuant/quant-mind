"""Preprocess layer — fetch + format two-stage data prep.

Imports are surfaced at the package root for the common path
(``from quantmind.preprocess import fetch_arxiv, pdf_to_markdown``) but
sub-modules remain available for callers that prefer the explicit path
(``from quantmind.preprocess.fetch.arxiv import fetch_arxiv``).
"""

from quantmind.preprocess.clean import (
    collapse_whitespace,
    dedupe_lines,
    normalize_unicode,
)
from quantmind.preprocess.fetch import (
    ArxivIdParseError,
    CrossrefMetadata,
    Fetched,
    RawPaper,
    fetch_arxiv,
    fetch_url,
    read_local_file,
    resolve_doi,
)
from quantmind.preprocess.format import (
    PdfParseError,
    html_to_markdown,
    pdf_to_markdown,
)
from quantmind.preprocess.time import (
    business_days_between,
    parse_filing_date,
    to_utc,
)

__all__ = [
    "ArxivIdParseError",
    "CrossrefMetadata",
    "Fetched",
    "PdfParseError",
    "RawPaper",
    "business_days_between",
    "collapse_whitespace",
    "dedupe_lines",
    "fetch_arxiv",
    "fetch_url",
    "html_to_markdown",
    "normalize_unicode",
    "parse_filing_date",
    "pdf_to_markdown",
    "read_local_file",
    "resolve_doi",
    "to_utc",
]
