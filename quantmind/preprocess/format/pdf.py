"""Page-aware PDF parsing."""

import asyncio
import hashlib
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from liteparse import LiteParse, ParseError


class PdfParseError(ValueError):
    """Raised when a PDF cannot be parsed into a complete page sequence."""


@dataclass(frozen=True)
class BoundingBox:
    """A rectangle in top-left-origin PDF page coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class TextBlock:
    """One parser-provided text block on a physical PDF page."""

    text: str
    page_number: int
    bbox: BoundingBox
    font_name: str | None = None
    font_size: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class ParsedPage:
    """One physical PDF page, including empty pages."""

    page_number: int
    width: float
    height: float
    text: str
    blocks: tuple[TextBlock, ...]
    screenshot_path: str | None = None
    image_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedDocument:
    """Deterministic page-aware result for one exact PDF byte stream."""

    source_hash: str
    parser_name: str
    parser_version: str
    cleanup_version: str
    pages: tuple[ParsedPage, ...]


def _write_artifacts(
    parser: LiteParse,
    pdf_bytes: bytes,
    artifact_dir: Path,
    images: list[Any],
) -> tuple[dict[int, str], dict[int, tuple[str, ...]]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = artifact_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    image_dir = artifact_dir / "images"
    image_dir.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
        source.write(pdf_bytes)
        source.flush()
        screenshots = parser.screenshot(source.name)
    screenshot_paths: dict[int, str] = {}
    for screenshot in screenshots:
        path = screenshots_dir / f"page_{screenshot.page_num}.png"
        path.write_bytes(screenshot.image_bytes)
        screenshot_paths[screenshot.page_num] = str(path.resolve())
    image_paths: dict[int, list[str]] = {}
    for image in images:
        path = image_dir / f"page_{image.page}_{image.id}.{image.format}"
        path.write_bytes(image.bytes)
        image_paths.setdefault(image.page, []).append(str(path.resolve()))
    return screenshot_paths, {
        page: tuple(paths) for page, paths in image_paths.items()
    }


def _parse_pdf_sync(
    pdf_bytes: bytes,
    artifact_dir: Path | None,
) -> ParsedDocument:
    parser = LiteParse(ocr_enabled=False, image_mode="embed", quiet=True)
    try:
        result = parser.parse(pdf_bytes)
        screenshots: dict[int, str] = {}
        images: dict[int, tuple[str, ...]] = {}
        if artifact_dir is not None:
            screenshots, images = _write_artifacts(
                parser, pdf_bytes, artifact_dir, result.images
            )
    except (ParseError, OSError, ValueError) as exc:
        raise PdfParseError(
            f"LiteParse could not parse PDF bytes: {exc}"
        ) from exc

    pages: list[ParsedPage] = []
    expected_page = 1
    for page in result.pages:
        if page.page_num != expected_page:
            raise PdfParseError(
                "LiteParse returned a non-contiguous physical page sequence"
            )
        blocks = tuple(
            TextBlock(
                text=item.text,
                page_number=page.page_num,
                bbox=BoundingBox(
                    x0=item.x,
                    y0=item.y,
                    x1=item.x + item.width,
                    y1=item.y + item.height,
                ),
                font_name=item.font_name,
                font_size=item.font_size,
                confidence=item.confidence,
            )
            for item in page.text_items
        )
        pages.append(
            ParsedPage(
                page_number=page.page_num,
                width=page.width,
                height=page.height,
                text=page.text,
                blocks=blocks,
                screenshot_path=screenshots.get(page.page_num),
                image_paths=images.get(page.page_num, ()),
            )
        )
        expected_page += 1
    if not pages:
        raise PdfParseError("LiteParse returned no physical pages")
    return ParsedDocument(
        source_hash=hashlib.sha256(pdf_bytes).hexdigest(),
        parser_name="liteparse",
        parser_version=version("liteparse"),
        cleanup_version="1",
        pages=tuple(pages),
    )


async def parse_pdf(
    pdf_bytes: bytes,
    *,
    artifact_dir: str | Path | None = None,
) -> ParsedDocument:
    """Parse exact PDF bytes while preserving physical pages and artifacts.

    Args:
        pdf_bytes: Exact bytes of one PDF source version.
        artifact_dir: Optional directory for page screenshots and extracted images.

    Returns:
        A page-aware deterministic document.

    Raises:
        PdfParseError: If input is empty, invalid, or has missing pages.
    """
    if not pdf_bytes:
        raise PdfParseError("pdf_bytes is empty")
    path = Path(artifact_dir).expanduser() if artifact_dir is not None else None
    return await asyncio.to_thread(_parse_pdf_sync, pdf_bytes, path)


async def pdf_to_markdown(pdf_bytes: bytes) -> str:
    """Return a compatibility text view derived from preserved PDF pages."""
    document = await parse_pdf(pdf_bytes)
    return "\n\n".join(page.text for page in document.pages)
