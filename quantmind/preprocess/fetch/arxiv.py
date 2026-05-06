"""ArXiv fetch helper.

Resolves an arXiv id (or full URL) to a ``RawPaper`` containing the PDF
bytes plus metadata pulled from arXiv's public API. Uses the existing
``arxiv`` Python lib (sync) for metadata via :func:`asyncio.to_thread`,
and ``httpx`` for the PDF download so the network legs are properly
async.
"""

import asyncio
import re
from datetime import datetime, timezone

import arxiv
import httpx

from quantmind.preprocess.fetch._types import RawPaper
from quantmind.preprocess.fetch.http import DEFAULT_USER_AGENT

# Accepts:
#   2401.12345
#   2401.12345v3
#   arXiv:2401.12345
#   http(s)://arxiv.org/abs/2401.12345
#   http(s)://arxiv.org/pdf/2401.12345v2.pdf
#   cs.AI/0102001 (legacy ID format)
_NEW_ID_PATTERN = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")
_LEGACY_ID_PATTERN = re.compile(r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?")


class ArxivIdParseError(ValueError):
    """Raised when an arXiv id/URL cannot be parsed."""


def _extract_arxiv_id(id_or_url: str) -> str:
    """Pull the canonical arXiv id from a raw user input.

    Returns:
        Canonical id in either modern (``YYMM.NNNNN``) or legacy
        (``archive[.subject]/YYMMNNN``) form, with version suffix preserved
        when present.

    Raises:
        ArxivIdParseError: If no recognizable arXiv id can be extracted.
    """
    candidate = id_or_url.strip()
    for prefix in ("arXiv:", "arxiv:"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
    match = _NEW_ID_PATTERN.search(candidate) or _LEGACY_ID_PATTERN.search(
        candidate
    )
    if match is None:
        raise ArxivIdParseError(
            f"could not parse arXiv id from input: {id_or_url!r}"
        )
    return match.group(0)


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fetch_metadata_sync(arxiv_id: str) -> arxiv.Result:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        raise LookupError(f"arXiv id not found: {arxiv_id!r}")
    return results[0]


async def fetch_arxiv(id_or_url: str) -> RawPaper:
    """Fetch arXiv metadata and PDF bytes for a single paper.

    Args:
        id_or_url: ArXiv id, ``arXiv:`` prefixed string, or full
            ``arxiv.org`` URL (abs or pdf form).

    Returns:
        ``RawPaper`` with PDF bytes, ``content_type='application/pdf'``,
        and metadata fields populated from the arXiv API response.

    Raises:
        ArxivIdParseError: If the id cannot be parsed.
        LookupError: If arXiv has no record for the parsed id.
        httpx.HTTPError: On PDF download failure.
    """
    arxiv_id = _extract_arxiv_id(id_or_url)
    result = await asyncio.to_thread(_fetch_metadata_sync, arxiv_id)

    pdf_url = result.pdf_url
    if not pdf_url:
        raise LookupError(f"arxiv result has no pdf_url for {arxiv_id!r}")
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(pdf_url, headers=headers)
        response.raise_for_status()
        pdf_bytes = response.content

    return RawPaper(
        bytes=pdf_bytes,
        content_type="application/pdf",
        source_url=pdf_url,
        headers={},
        arxiv_id=arxiv_id,
        title=result.title,
        authors=tuple(str(a) for a in result.authors),
        abstract=result.summary,
        published_at=_to_utc(result.published),
        primary_category=result.primary_category,
        categories=tuple(str(c) for c in result.categories),
    )
