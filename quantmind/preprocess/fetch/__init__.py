"""Fetch layer — pulls raw bytes/metadata from external sources.

Every function is ``async def`` and returns a frozen dataclass. No LLM
calls, no parsing — that's the format layer's job.
"""

from quantmind.preprocess.fetch._types import Fetched, RawPaper
from quantmind.preprocess.fetch.arxiv import (
    ArxivIdParseError,
    fetch_arxiv,
)
from quantmind.preprocess.fetch.doi import (
    CrossrefMetadata,
    resolve_doi,
)
from quantmind.preprocess.fetch.http import (
    DEFAULT_USER_AGENT,
    fetch_url,
)
from quantmind.preprocess.fetch.local import read_local_file

__all__ = [
    "ArxivIdParseError",
    "CrossrefMetadata",
    "DEFAULT_USER_AGENT",
    "Fetched",
    "RawPaper",
    "fetch_arxiv",
    "fetch_url",
    "read_local_file",
    "resolve_doi",
]
