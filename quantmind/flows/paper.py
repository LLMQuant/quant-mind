"""Source-first paper flow that returns chunks before a cited summary."""

import hashlib
import mimetypes
from dataclasses import dataclass
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
    ArtifactLocator,
    PaperAssetRef,
    PaperBoundingBox,
    PaperChunk,
    PaperChunkingConfig,
    PaperChunkSet,
    PaperCitation,
    PaperFlowResult,
    PaperGlobalSummary,
    PaperParsedBlock,
    PaperParsedManifest,
    PaperParsedPage,
    PaperSourceRevision,
    PaperSourceSpan,
    PaperSummaryProducer,
    SourceRef,
)
from quantmind.knowledge.paper import (
    _paper_artifact_id,
    _paper_asset_id,
    _paper_chunk_id,
    _paper_chunk_set_content_hash,
    _paper_source_id,
    _paper_summary_content_hash,
    _stable_hash,
    _text_hash,
)
from quantmind.preprocess.fetch import (
    Fetched,
    RawPaper,
    fetch_arxiv,
    fetch_url,
    read_local_file,
)
from quantmind.preprocess.format import ParsedDocument, parse_pdf
from quantmind.rag import SentenceSplitterConfig, chunk_parsed_document


class UnsupportedContentTypeError(ValueError):
    """The source is not a page-aware PDF supported by Paper Flow V1."""


class PaperCitationValidationError(ValueError):
    """A generated summary did not provide valid source coverage."""


@dataclass(frozen=True)
class _FetchedPaperSource:
    """Exact fetched bytes and code-owned source facts before parsing."""

    bytes: bytes
    media_type: str
    kind: Literal["arxiv", "http", "local"]
    uri: str
    fetched_at: datetime
    available_at: datetime
    published_at: datetime | None = None
    arxiv_id: str | None = None
    title: str | None = None
    authors: tuple[str, ...] = ()


async def paper_flow(
    input: PaperInput,
    *,
    cfg: PaperFlowCfg | None = None,
    _summary_provider: _PaperSummaryProvider | None = None,
) -> PaperFlowResult:
    """Build a page-aware chunk set and one cited global summary.

    IDs, source metadata, artifact membership, lineage, and citation links are
    created and validated by code. The model returns only summary prose and
    chunk/page coordinates through a bounded summarization seam.

    Args:
        input: Typed paper source. V1 requires a PDF-backed input.
        cfg: Splitter, summary model, and explicit usage/runtime limits.

    Returns:
        The exact source revision, one chunk-set artifact, and one cited
        global-summary artifact.

    Raises:
        UnsupportedContentTypeError: If the resolved content is not a PDF.
        PaperCitationValidationError: If generated citations are invalid or
            do not meet configured source-coverage requirements.
        NotImplementedError: If a DOI input has no exact open PDF resolver.
    """
    cfg = cfg or PaperFlowCfg()
    fetched = await _fetch_paper_source(input)
    parsed = await parse_pdf(
        fetched.bytes,
        artifact_dir=cfg.output_dir,
    )
    source = _build_source_revision(fetched, parsed)
    parsed_chunks = chunk_parsed_document(
        parsed,
        config=SentenceSplitterConfig(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        ),
    )
    chunk_set = _build_chunk_set(source, parsed_chunks, cfg)
    provider = _summary_provider or _AgentsPaperSummaryProvider()
    draft = await provider.summarize(source, chunk_set, cfg=cfg)
    summary = _build_global_summary(source, chunk_set, draft, cfg)
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


async def _fetch_paper_source(input: PaperInput) -> _FetchedPaperSource:
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
        return _FetchedPaperSource(
            bytes=raw_paper.bytes,
            media_type="application/pdf",
            kind="arxiv",
            uri=uri,
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
        return _FetchedPaperSource(
            bytes=raw_http.bytes,
            media_type="application/pdf",
            kind="http",
            uri=raw_http.resolved_url or raw_http.source_url or input.url,
            fetched_at=fetched_at,
            available_at=fetched_at,
        )
    if isinstance(input, LocalFilePath):
        raw_local = await read_local_file(input.path)
        _require_pdf(raw_local)
        observed_at = _aware_or_now(raw_local.fetched_at)
        path = Path(input.path).expanduser().resolve()
        return _FetchedPaperSource(
            bytes=raw_local.bytes,
            media_type="application/pdf",
            kind="local",
            uri=raw_local.source_url or path.as_uri(),
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


def _asset_from_path(
    path_value: str,
    *,
    source_revision_id,
    kind: Literal["screenshot", "image"],
    page_number: int,
    assets: dict,
    blobs: dict[str, bytes],
) -> PaperAssetRef:
    path = Path(path_value)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"Parser asset for page {page_number} is missing: {path}"
        ) from exc
    content_hash = hashlib.sha256(content).hexdigest()
    media_type = (
        mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )
    asset = PaperAssetRef(
        asset_id=_paper_asset_id(
            source_revision_id,
            kind=kind,
            page_number=page_number,
            content_hash=content_hash,
        ),
        kind=kind,
        media_type=media_type,
        content_hash=content_hash,
        size_bytes=len(content),
        page_number=page_number,
    )
    assets[asset.asset_id] = asset
    blobs[content_hash] = content
    return asset


def _build_source_revision(
    fetched: _FetchedPaperSource,
    parsed: ParsedDocument,
) -> PaperSourceRevision:
    source_id = _paper_source_id(parsed.source_hash)
    blobs = {parsed.source_hash: fetched.bytes}
    raw = PaperAssetRef(
        asset_id=_paper_asset_id(
            source_id,
            kind="raw",
            page_number=None,
            content_hash=parsed.source_hash,
        ),
        kind="raw",
        media_type=fetched.media_type,
        content_hash=parsed.source_hash,
        size_bytes=len(fetched.bytes),
    )
    assets = {raw.asset_id: raw}
    pages: list[PaperParsedPage] = []
    for page in parsed.pages:
        screenshot = (
            _asset_from_path(
                page.screenshot_path,
                source_revision_id=source_id,
                kind="screenshot",
                page_number=page.page_number,
                assets=assets,
                blobs=blobs,
            )
            if page.screenshot_path is not None
            else None
        )
        images = tuple(
            _asset_from_path(
                image_path,
                source_revision_id=source_id,
                kind="image",
                page_number=page.page_number,
                assets=assets,
                blobs=blobs,
            )
            for image_path in page.image_paths
        )
        pages.append(
            PaperParsedPage(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                text=page.text,
                blocks=tuple(
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
                ),
                screenshot_asset_id=(
                    screenshot.asset_id if screenshot is not None else None
                ),
                image_asset_ids=tuple(image.asset_id for image in images),
            )
        )
    source_ref = SourceRef(
        kind=fetched.kind,
        uri=fetched.uri,
        fetched_at=fetched.fetched_at,
        content_hash=parsed.source_hash,
    )
    return PaperSourceRevision(
        id=source_id,
        source=source_ref,
        as_of=fetched.published_at or fetched.available_at,
        available_at=fetched.available_at,
        published_at=fetched.published_at,
        arxiv_id=fetched.arxiv_id,
        title=fetched.title,
        authors=fetched.authors,
        parsed=PaperParsedManifest(
            source_hash=parsed.source_hash,
            parser_name=parsed.parser_name,
            parser_version=parsed.parser_version,
            cleanup_version=parsed.cleanup_version,
            pages=tuple(pages),
        ),
        raw_asset_id=raw.asset_id,
        assets=tuple(assets.values()),
        blobs=blobs,
    )


def _build_chunk_set(
    source: PaperSourceRevision,
    parsed_chunks,
    cfg: PaperFlowCfg,
) -> PaperChunkSet:
    producer = PaperChunkingConfig(
        splitter_version=version("llama-index-core"),
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    producer_hash = _stable_hash(producer.model_dump(mode="json"))
    artifact_id = _paper_artifact_id(
        source.id,
        "paper_chunk_set",
        producer_hash,
    )
    page_assets = {
        page.page_number: tuple(
            asset_id
            for asset_id in (
                (page.screenshot_asset_id,)
                if page.screenshot_asset_id is not None
                else ()
            )
            + page.image_asset_ids
        )
        for page in source.parsed.pages
    }
    chunks: list[PaperChunk] = []
    for position, parsed_chunk in enumerate(parsed_chunks):
        if parsed_chunk.source_hash != source.source.content_hash:
            raise ValueError("parsed chunk belongs to another source revision")
        span = PaperSourceSpan(
            page_number=parsed_chunk.page_number,
            start_char=parsed_chunk.start_char,
            end_char=parsed_chunk.end_char,
            block_boxes=tuple(
                PaperBoundingBox(
                    x0=box.x0,
                    y0=box.y0,
                    x1=box.x1,
                    y1=box.y1,
                )
                for box in parsed_chunk.block_boxes
            ),
            asset_ids=page_assets.get(parsed_chunk.page_number, ()),
        )
        content_hash = _text_hash(parsed_chunk.text)
        chunks.append(
            PaperChunk(
                chunk_id=_paper_chunk_id(
                    artifact_id,
                    position=position,
                    content_hash=content_hash,
                    spans=(span,),
                ),
                chunk_set_id=artifact_id,
                source_revision_id=source.id,
                position=position,
                text=parsed_chunk.text,
                content_hash=content_hash,
                source_spans=(span,),
            )
        )
    if not chunks:
        raise ValueError("paper source produced no non-empty chunks")
    chunk_tuple = tuple(chunks)
    return PaperChunkSet(
        id=artifact_id,
        source_revision_id=source.id,
        producer=producer,
        producer_config_hash=producer_hash,
        content_hash=_paper_chunk_set_content_hash(chunk_tuple),
        chunks=chunk_tuple,
    )


def _build_global_summary(
    source: PaperSourceRevision,
    chunk_set: PaperChunkSet,
    draft: PaperSummaryDraft,
    cfg: PaperFlowCfg,
) -> PaperGlobalSummary:
    citations: list[PaperCitation] = []
    seen: set[tuple[int, int, str | None]] = set()
    for draft_citation in draft.citations:
        try:
            chunk = chunk_set.chunks[draft_citation.chunk_index]
        except IndexError as exc:
            raise PaperCitationValidationError(
                "paper summary cites an unknown chunk index"
            ) from exc
        pages = {span.page_number for span in chunk.source_spans}
        if draft_citation.page_number not in pages:
            raise PaperCitationValidationError(
                "paper summary citation page is not owned by its chunk"
            )
        quote = draft_citation.quote
        if quote is not None and quote not in chunk.text:
            raise PaperCitationValidationError(
                "paper summary citation quote is not present in its chunk"
            )
        key = (draft_citation.chunk_index, draft_citation.page_number, quote)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            PaperCitation(
                chunk_set_id=chunk_set.id,
                chunk_id=chunk.chunk_id,
                page_number=draft_citation.page_number,
                quote=quote,
            )
        )
    cited_pages = {citation.page_number for citation in citations}
    if len(citations) < cfg.min_summary_citations:
        raise PaperCitationValidationError(
            "paper summary has fewer citations than min_summary_citations"
        )
    if len(cited_pages) < cfg.min_summary_pages:
        raise PaperCitationValidationError(
            "paper summary has fewer source pages than min_summary_pages"
        )

    producer = PaperSummaryProducer(
        model=cfg.model,
        prompt_version=cfg.summary_prompt_version,
        input_chunk_set_id=chunk_set.id,
        instructions_hash=_summary_instructions_hash(cfg),
        max_output_tokens=cfg.max_summary_output_tokens,
    )
    producer_hash = _stable_hash(producer.model_dump(mode="json"))
    artifact_id = _paper_artifact_id(
        source.id,
        "paper_summary",
        producer_hash,
    )
    citation_tuple = tuple(citations)
    summary_text = draft.summary.strip()
    return PaperGlobalSummary(
        id=artifact_id,
        source_revision_id=source.id,
        producer=producer,
        producer_config_hash=producer_hash,
        content_hash=_paper_summary_content_hash(
            summary_text,
            citation_tuple,
        ),
        summary=summary_text,
        citations=citation_tuple,
        derived_from=(
            ArtifactLocator(
                source_revision_id=source.id,
                artifact_id=chunk_set.id,
                artifact_kind="paper_chunk_set",
            ),
        ),
    )
