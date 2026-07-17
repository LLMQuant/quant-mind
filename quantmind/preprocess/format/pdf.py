"""Page-aware PDF parsing and private LlamaIndex ingestion."""

import asyncio
import hashlib
import json
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from liteparse import LiteParse, ParseError
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.retrievers.bm25 import BM25Retriever


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


@dataclass(frozen=True)
class SentenceSplitterConfig:
    """Supported LlamaIndex sentence-splitting parameters."""

    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass(frozen=True)
class ParsedChunk:
    """QuantMind view of a private LlamaIndex text node."""

    chunk_id: str
    text: str
    source_hash: str
    page_number: int
    block_boxes: tuple[BoundingBox, ...]
    screenshot_path: str | None
    image_paths: tuple[str, ...]


@dataclass(frozen=True)
class ParsedDocumentHit:
    """Ranked page-aware evidence returned from document retrieval."""

    chunk: ParsedChunk
    score: float


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


def _page_metadata(
    document: ParsedDocument, page: ParsedPage
) -> dict[str, Any]:
    return {
        "source_hash": document.source_hash,
        "page_number": page.page_number,
        "block_boxes": json.dumps(
            [
                [block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1]
                for block in page.blocks
            ],
            separators=(",", ":"),
        ),
        "screenshot_path": page.screenshot_path or "",
        "image_paths": json.dumps(page.image_paths, separators=(",", ":")),
    }


def _to_llama_documents(document: ParsedDocument) -> list[Document]:
    return [
        Document(
            text=page.text,
            id_=f"{document.source_hash}:page:{page.page_number}",
            metadata=_page_metadata(document, page),
            excluded_embed_metadata_keys=[
                "block_boxes",
                "screenshot_path",
                "image_paths",
            ],
            excluded_llm_metadata_keys=["block_boxes"],
        )
        for page in document.pages
        if page.text.strip()
    ]


def _node_to_chunk(node: BaseNode) -> ParsedChunk:
    metadata = node.metadata
    boxes = tuple(
        BoundingBox(*values) for values in json.loads(metadata["block_boxes"])
    )
    return ParsedChunk(
        chunk_id=node.node_id,
        text=node.get_content(metadata_mode=MetadataMode.NONE),
        source_hash=str(metadata["source_hash"]),
        page_number=int(metadata["page_number"]),
        block_boxes=boxes,
        screenshot_path=str(metadata["screenshot_path"]) or None,
        image_paths=tuple(json.loads(metadata["image_paths"])),
    )


def chunk_parsed_document(
    document: ParsedDocument,
    *,
    config: SentenceSplitterConfig | None = None,
) -> tuple[ParsedChunk, ...]:
    """Split preserved PDF pages with LlamaIndex `SentenceSplitter`."""
    config = config or SentenceSplitterConfig()
    splitter = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(_to_llama_documents(document))
    return tuple(_node_to_chunk(node) for node in nodes)


def retrieve_parsed_document(
    chunks: tuple[ParsedChunk, ...],
    query: str,
    *,
    top_k: int = 5,
) -> tuple[ParsedDocumentHit, ...]:
    """Rank parsed chunks with the private LlamaIndex BM25 retriever."""
    if not query.strip():
        raise ValueError("query must not be blank")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not chunks:
        return ()
    nodes: list[BaseNode] = [
        TextNode(
            id_=chunk.chunk_id,
            text=chunk.text,
            metadata={"chunk_index": index},
        )
        for index, chunk in enumerate(chunks)
    ]
    retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=min(top_k, len(nodes)),
    )
    results = retriever.retrieve(query)
    return tuple(
        ParsedDocumentHit(
            chunk=chunks[int(result.node.metadata["chunk_index"])],
            score=float(result.score or 0.0),
        )
        for result in results
    )


async def pdf_to_markdown(pdf_bytes: bytes) -> str:
    """Return a compatibility text view derived from preserved PDF pages."""
    document = await parse_pdf(pdf_bytes)
    return "\n\n".join(page.text for page in document.pages)
