"""Source-first paper flow that returns chunks before a cited summary.

The flow is deliberately thin. It owns only what genuinely needs both the
preprocess/rag value objects and IO: fetching bytes, parsing the PDF, reading
page-asset bytes off disk, and mapping those ephemeral, path-based artifacts
into knowledge-native inputs. All identity (IDs, content and producer hashes,
citation resolution) lives on the knowledge models' ``from_*`` constructors,
so this module imports no private ID helpers and computes no paper ID itself.
"""

import mimetypes
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import (
    ArxivIdentifier,
    DoiIdentifier,
    HttpUrl,
    LocalFilePath,
    PaperInput,
    RawText,
)
from quantmind.flows._paper_summary import (
    PaperSummaryDraft,
    _AgentsPaperSummaryProvider,
    _PaperSummaryProvider,
    _summary_instructions_hash,
)
from quantmind.knowledge import (
    PaperAssetInput,
    PaperBoundingBox,
    PaperChunkingConfig,
    PaperChunkInput,
    PaperChunkSet,
    PaperCitationDraft,
    PaperFlowResult,
    PaperGlobalSummary,
    PaperPageInput,
    PaperParsedBlock,
    PaperSourceFacts,
    PaperSourceRevision,
    PaperSummaryProducer,
)
from quantmind.preprocess.fetch import (
    Fetched,
    RawPaper,
    fetch_arxiv,
    fetch_url,
    read_local_file,
)
from quantmind.preprocess.format import ParsedDocument, parse_pdf
from quantmind.rag import (
    ParsedChunk,
    SentenceSplitterConfig,
    chunk_parsed_document,
)


class UnsupportedContentTypeError(ValueError):
    """The source is not a page-aware PDF supported by Paper Flow V1."""


async def paper_flow(
    input: PaperInput,
    *,
    cfg: PaperFlowCfg | None = None,
    _summary_provider: _PaperSummaryProvider | None = None,
) -> PaperFlowResult:
    """Build a page-aware chunk set and one cited global summary.

    IDs, source metadata, artifact membership, lineage, and citation links are
    minted and validated by the knowledge-layer constructors. The model returns
    only summary prose and chunk/page coordinates through a bounded seam.

    Args:
        input: Typed paper source. V1 requires a PDF-backed input.
        cfg: Splitter, summary model, and usage/runtime limits.

    Returns:
        The exact source revision, one chunk-set artifact, and one cited
        global-summary artifact.

    Raises:
        UnsupportedContentTypeError: If the resolved content is not a PDF.
        PaperCitationValidationError: If generated citations are invalid or do
            not meet the configured source-coverage policy.
        NotImplementedError: If a DOI input has no exact open PDF resolver.
    """
    cfg = cfg or PaperFlowCfg()
    facts = await _fetch_paper_source(input)
    parsed = await parse_pdf(facts.raw_bytes, artifact_dir=cfg.output_dir)
    source = PaperSourceRevision.from_parsed(
        facts=facts,
        source_hash=parsed.source_hash,
        parser_name=parsed.parser_name,
        parser_version=parsed.parser_version,
        cleanup_version=parsed.cleanup_version,
        pages=_adapt_pages(parsed),
    )
    parsed_chunks = chunk_parsed_document(
        parsed,
        config=SentenceSplitterConfig(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        ),
    )
    producer = PaperChunkingConfig(
        splitter_version=version("llama-index-core"),
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunk_set = PaperChunkSet.from_parsed_chunks(
        source,
        _adapt_chunks(parsed_chunks, source),
        producer=producer,
    )
    provider = _summary_provider or _AgentsPaperSummaryProvider()
    draft = await provider.summarize(source, chunk_set, cfg=cfg)
    summary = _build_summary(chunk_set, draft, cfg)
    return PaperFlowResult(
        source_revision=source,
        chunk_set=chunk_set,
        global_summary=summary,
    )


def _aware_or_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _require_pdf(raw: Fetched) -> None:
    media_type = (raw.content_type or "").lower()
    if not media_type.startswith("application/pdf"):
        raise UnsupportedContentTypeError(
            "Paper Flow V1 requires a page-aware PDF; resolved content type "
            f"was {media_type!r}"
        )


async def _fetch_paper_source(input: PaperInput) -> PaperSourceFacts:
    """Fetch and normalize one input into code-owned source facts (IO)."""
    if isinstance(input, ArxivIdentifier):
        raw_paper: RawPaper = await fetch_arxiv(input.id)
        _require_pdf(raw_paper)
        fetched_at = _aware_or_now(raw_paper.fetched_at)
        available_at = _aware_or_now(
            raw_paper.updated_at or raw_paper.published_at or fetched_at
        )
        uri = raw_paper.resolved_url or raw_paper.source_url
        if uri is None:
            raise ValueError("resolved arXiv paper is missing its source URL")
        return PaperSourceFacts(
            kind="arxiv",
            uri=uri,
            media_type="application/pdf",
            raw_bytes=raw_paper.bytes,
            fetched_at=fetched_at,
            available_at=available_at,
            published_at=raw_paper.published_at,
            arxiv_id=raw_paper.arxiv_id,
            title=raw_paper.title,
            authors=raw_paper.authors,
        )
    if isinstance(input, HttpUrl):
        raw_http = await fetch_url(input.url)
        _require_pdf(raw_http)
        fetched_at = _aware_or_now(raw_http.fetched_at)
        return PaperSourceFacts(
            kind="http",
            uri=raw_http.resolved_url or raw_http.source_url or input.url,
            media_type="application/pdf",
            raw_bytes=raw_http.bytes,
            fetched_at=fetched_at,
            available_at=fetched_at,
        )
    if isinstance(input, LocalFilePath):
        raw_local = await read_local_file(input.path)
        _require_pdf(raw_local)
        observed_at = _aware_or_now(raw_local.fetched_at)
        path = Path(input.path).expanduser().resolve()
        return PaperSourceFacts(
            kind="local",
            uri=raw_local.source_url or path.as_uri(),
            media_type="application/pdf",
            raw_bytes=raw_local.bytes,
            fetched_at=observed_at,
            available_at=observed_at,
        )
    if isinstance(input, RawText):
        raise UnsupportedContentTypeError(
            "Paper Flow V1 requires a page-aware PDF; RawText has no physical "
            "page evidence"
        )
    if isinstance(input, DoiIdentifier):
        raise NotImplementedError(
            "DOI inputs require an exact open PDF resolver before they can "
            "produce a paper source revision"
        )
    raise TypeError(f"Unsupported PaperInput variant: {type(input)!r}")


def _read_page_asset(
    path_value: str,
    *,
    kind: Literal["screenshot", "image"],
    page_number: int,
) -> PaperAssetInput:
    """Read one parser-written page asset off disk into knowledge input (IO)."""
    path = Path(path_value)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"Parser asset for page {page_number} is missing: {path}"
        ) from exc
    media_type = (
        mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )
    return PaperAssetInput(kind=kind, content=content, media_type=media_type)


def _adapt_pages(parsed: ParsedDocument) -> list[PaperPageInput]:
    """Map path-based parsed pages into knowledge-native page inputs."""
    pages: list[PaperPageInput] = []
    for page in parsed.pages:
        blocks = tuple(
            PaperParsedBlock(
                text=block.text,
                bbox=PaperBoundingBox(
                    x0=block.bbox.x0,
                    y0=block.bbox.y0,
                    x1=block.bbox.x1,
                    y1=block.bbox.y1,
                ),
                font_name=block.font_name,
                font_size=block.font_size,
                confidence=block.confidence,
            )
            for block in page.blocks
        )
        screenshot = (
            _read_page_asset(
                page.screenshot_path,
                kind="screenshot",
                page_number=page.page_number,
            )
            if page.screenshot_path is not None
            else None
        )
        images = tuple(
            _read_page_asset(
                image_path,
                kind="image",
                page_number=page.page_number,
            )
            for image_path in page.image_paths
        )
        pages.append(
            PaperPageInput(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                text=page.text,
                blocks=blocks,
                screenshot=screenshot,
                images=images,
            )
        )
    return pages


def _adapt_chunks(
    parsed_chunks: Sequence[ParsedChunk],
    source: PaperSourceRevision,
) -> list[PaperChunkInput]:
    """Map splitter chunks into knowledge-native chunk inputs."""
    content_hash = source.source.content_hash
    chunks: list[PaperChunkInput] = []
    for parsed_chunk in parsed_chunks:
        if parsed_chunk.source_hash != content_hash:
            raise ValueError("parsed chunk belongs to another source revision")
        chunks.append(
            PaperChunkInput(
                page_number=parsed_chunk.page_number,
                start_char=parsed_chunk.start_char,
                end_char=parsed_chunk.end_char,
                block_boxes=tuple(
                    PaperBoundingBox(x0=box.x0, y0=box.y0, x1=box.x1, y1=box.y1)
                    for box in parsed_chunk.block_boxes
                ),
                text=parsed_chunk.text,
            )
        )
    return chunks


def _build_summary(
    chunk_set: PaperChunkSet,
    draft: PaperSummaryDraft,
    cfg: PaperFlowCfg,
) -> PaperGlobalSummary:
    """Assemble the summary producer and delegate identity to the model."""
    producer = PaperSummaryProducer(
        model=cfg.model,
        prompt_version=cfg.summary_prompt_version,
        input_chunk_set_id=chunk_set.id,
        instructions_hash=_summary_instructions_hash(cfg),
        max_output_tokens=cfg.max_summary_output_tokens,
        research_group_size=cfg.summary_research_group_size,
    )
    citations = tuple(
        PaperCitationDraft(
            chunk_index=citation.chunk_index,
            page_number=citation.page_number,
            quote=citation.quote,
        )
        for citation in draft.citations
    )
    return PaperGlobalSummary.from_draft(
        chunk_set,
        producer=producer,
        summary=draft.summary,
        citations=citations,
        min_citations=cfg.min_summary_citations,
        min_pages=cfg.min_summary_pages,
    )
