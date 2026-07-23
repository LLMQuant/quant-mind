"""Deterministic outline signals from page-aware parsed documents."""

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median

from quantmind.preprocess.format import ParsedDocument, ParsedPage, TextBlock

_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+\S")
_TOC_ENTRY = re.compile(r"^.{2,}?\.{2,}\s*\d+\s*$")
_STANDALONE_PAGE = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,8})\s*$", re.I)


@dataclass(frozen=True)
class HeadingCandidate:
    """One ordered, physical-page-anchored heading signal."""

    page_number: int
    text: str
    level_hint: int
    font_size: float | None = None


@dataclass(frozen=True)
class OutlineSignals:
    """Deterministic signals supplied to a semantic structuring stage."""

    table_of_contents_pages: tuple[int, ...]
    headings: tuple[HeadingCandidate, ...]
    printed_page_offset: int | None


def extract_outline_signals(document: ParsedDocument) -> OutlineSignals:
    """Extract table-of-contents, heading, and page-offset hints.

    The operation uses only parser-owned text, block coordinates, and font
    sizes. It performs no model call and preserves physical page numbering.

    Args:
        document: One exact, page-aware parsed PDF.

    Returns:
        Ordered outline hints suitable for a later structuring model call.
    """
    body_size = _body_font_size(document)
    repeated_headers = _repeated_header_text(document)
    headings = tuple(
        candidate
        for page in document.pages
        for block in page.blocks
        if (
            candidate := _heading_candidate(
                page,
                block,
                body_size=body_size,
                repeated_headers=repeated_headers,
            )
        )
        is not None
    )
    return OutlineSignals(
        table_of_contents_pages=tuple(
            page.page_number
            for page in document.pages
            if _is_table_of_contents(page)
        ),
        headings=headings,
        printed_page_offset=_printed_page_offset(document),
    )


def _body_font_size(document: ParsedDocument) -> float | None:
    sizes = [
        block.font_size
        for page in document.pages
        for block in page.blocks
        if block.font_size is not None
        and block.text.strip()
        and len(block.text.strip()) >= 20
    ]
    return float(median(sizes)) if sizes else None


def _repeated_header_text(document: ParsedDocument) -> set[str]:
    pages_by_text: dict[str, set[int]] = {}
    for page in document.pages:
        for block in page.blocks:
            if block.bbox.y0 > page.height * 0.12:
                continue
            text = " ".join(block.text.split()).casefold()
            if text:
                pages_by_text.setdefault(text, set()).add(page.page_number)
    return {
        text
        for text, pages in pages_by_text.items()
        if len(pages) >= 2 and len(pages) * 2 >= len(document.pages)
    }


def _heading_candidate(
    page: ParsedPage,
    block: TextBlock,
    *,
    body_size: float | None,
    repeated_headers: set[str],
) -> HeadingCandidate | None:
    text = " ".join(block.text.split())
    if not text or len(text) > 120:
        return None
    if text.casefold() in repeated_headers:
        return None
    numbered = _NUMBERED_HEADING.match(text)
    prominent = (
        block.font_size is not None
        and body_size is not None
        and block.font_size >= body_size + 1.5
    )
    uppercase = (
        len(text) >= 4
        and any(character.isalpha() for character in text)
        and text.upper() == text
        and len(text.split()) <= 12
    )
    if numbered is None and not prominent and not uppercase:
        return None
    if numbered is not None:
        level_hint = numbered.group(1).count(".") + 1
    elif block.font_size is not None and body_size is not None:
        level_hint = 1 if block.font_size >= body_size + 4 else 2
    else:
        level_hint = 1
    return HeadingCandidate(
        page_number=page.page_number,
        text=text,
        level_hint=level_hint,
        font_size=block.font_size,
    )


def _is_table_of_contents(page: ParsedPage) -> bool:
    lines = [" ".join(line.split()) for line in page.text.splitlines()]
    normalized = {line.casefold() for line in lines if line}
    has_title = bool(normalized & {"contents", "table of contents"})
    return (
        has_title and sum(bool(_TOC_ENTRY.match(line)) for line in lines) >= 2
    )


def _printed_page_offset(document: ParsedDocument) -> int | None:
    offsets: list[int] = []
    for page in document.pages:
        candidates = [
            block.text
            for block in page.blocks
            if block.bbox.y0 >= page.height * 0.8
        ]
        for value in candidates:
            match = _STANDALONE_PAGE.match(value)
            if match is None:
                continue
            printed = _parse_printed_page(match.group(1))
            if printed is not None:
                offsets.append(page.page_number - printed)
                break
    if not offsets:
        return None
    counts = Counter(offsets)
    offset, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return offset if count >= 2 else None


def _parse_printed_page(value: str) -> int | None:
    if value.isdigit():
        parsed = int(value)
        return parsed if parsed >= 1 else None
    roman_values = {
        "i": 1,
        "v": 5,
        "x": 10,
        "l": 50,
        "c": 100,
        "d": 500,
        "m": 1000,
    }
    total = 0
    previous = 0
    for character in reversed(value.casefold()):
        current = roman_values[character]
        total += -current if current < previous else current
        previous = max(previous, current)
    return total or None
