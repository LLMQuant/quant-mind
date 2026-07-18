"""Page-aware document chunking and retrieval through LlamaIndex."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from quantmind.preprocess.format import (
    BoundingBox,
    ParsedDocument,
    ParsedPage,
)


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
    start_char: int
    end_char: int
    block_boxes: tuple[BoundingBox, ...]
    screenshot_path: str | None
    image_paths: tuple[str, ...]


@dataclass(frozen=True)
class ParsedDocumentHit:
    """Ranked page-aware evidence returned from document retrieval."""

    chunk: ParsedChunk
    score: float


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
    text = node.get_content(metadata_mode=MetadataMode.NONE)
    start_value = getattr(node, "start_char_idx", None)
    end_value = getattr(node, "end_char_idx", None)
    start_char = int(start_value) if start_value is not None else 0
    end_char = (
        int(end_value) if end_value is not None else start_char + len(text)
    )
    identity = hashlib.sha256(
        (
            f"{metadata['source_hash']}:{metadata['page_number']}:"
            f"{start_char}:{end_char}:{text}"
        ).encode("utf-8")
    ).hexdigest()
    boxes = tuple(
        BoundingBox(*values) for values in json.loads(metadata["block_boxes"])
    )
    return ParsedChunk(
        chunk_id=identity,
        text=text,
        source_hash=str(metadata["source_hash"]),
        page_number=int(metadata["page_number"]),
        start_char=start_char,
        end_char=end_char,
        block_boxes=boxes,
        screenshot_path=str(metadata["screenshot_path"]) or None,
        image_paths=tuple(json.loads(metadata["image_paths"])),
    )


def chunk_parsed_document(
    document: ParsedDocument,
    *,
    config: SentenceSplitterConfig | None = None,
) -> tuple[ParsedChunk, ...]:
    """Split preserved document pages with LlamaIndex `SentenceSplitter`."""
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
    """Rank parsed chunks with the opinionated LlamaIndex BM25 retriever."""
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
