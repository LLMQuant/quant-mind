"""DOI resolver via Crossref's open API.

Resolves a DOI to canonical metadata (title, authors, journal, publisher,
publication date, primary URL). The primary URL points at the publisher's
landing page — it is *not* guaranteed to be a direct PDF link. For OA PDF
discovery, see follow-up issue "Add unpaywall fallback to fetch/doi.py".
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

CROSSREF_BASE_URL = "https://api.crossref.org/works"

# Crossref accepts DOIs in their canonical form (10.NNNN/...). We accept a
# few common decorations users paste in (URL prefix, "doi:" prefix) and
# normalize before sending.
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")


@dataclass(frozen=True, slots=True)
class CrossrefMetadata:
    """Subset of Crossref's ``works/{doi}`` response we surface to callers."""

    doi: str
    title: str | None
    authors: tuple[str, ...]
    journal: str | None
    publisher: str | None
    published_at: datetime | None
    primary_url: str | None


def _normalize_doi(raw: str) -> str:
    """Strip common decorations from a user-supplied DOI string."""
    cleaned = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "DOI:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned


def _parse_crossref_date(parts: list[list[int]] | None) -> datetime | None:
    """Convert Crossref's ``date-parts: [[YYYY, MM, DD]]`` to a UTC datetime.

    Crossref returns date-parts arrays where missing components are simply
    omitted (``[[2024]]`` for year-only, ``[[2024, 5]]`` for year+month).
    Default missing month/day to January 1.
    """
    if not parts or not parts[0]:
        return None
    components = parts[0]
    year = components[0]
    month = components[1] if len(components) > 1 else 1
    day = components[2] if len(components) > 2 else 1
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _format_authors(items: list[dict[str, str]] | None) -> tuple[str, ...]:
    if not items:
        return ()
    out: list[str] = []
    for entry in items:
        given = entry.get("given", "").strip()
        family = entry.get("family", "").strip()
        if given and family:
            out.append(f"{given} {family}")
        elif family:
            out.append(family)
        elif given:
            out.append(given)
    return tuple(out)


async def resolve_doi(
    doi: str,
    *,
    timeout: float = 15.0,
) -> CrossrefMetadata:
    """Look up a DOI on Crossref and return canonical metadata.

    Args:
        doi: A DOI string. Accepts canonical form (``10.NNNN/...``),
            ``doi:`` prefix, or ``https://doi.org/`` URL form.
        timeout: HTTP timeout in seconds.

    Returns:
        ``CrossrefMetadata`` populated from the ``message`` block of
        Crossref's response.

    Raises:
        ValueError: If the DOI is malformed.
        httpx.HTTPStatusError: On 404 (DOI not registered) or 5xx.
    """
    normalized = _normalize_doi(doi)
    if not _DOI_PATTERN.match(normalized):
        raise ValueError(f"malformed DOI: {doi!r}")

    url = f"{CROSSREF_BASE_URL}/{normalized}"
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        body = response.json()

    msg = body.get("message", {})
    titles = msg.get("title") or []
    container = msg.get("container-title") or []
    issued = msg.get("issued", {}).get("date-parts")

    return CrossrefMetadata(
        doi=normalized,
        title=titles[0] if titles else None,
        authors=_format_authors(msg.get("author")),
        journal=container[0] if container else None,
        publisher=msg.get("publisher"),
        published_at=_parse_crossref_date(issued),
        primary_url=msg.get("URL"),
    )
