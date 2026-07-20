"""PaperFlow — a document-scoped handle grouping the paper pipelines.

``PaperFlow`` binds one immutable parsed source at ``open`` (fetch + parse run
exactly once) and exposes the finished paper pipelines over it:

- ``build_structure`` returns a self-contained ``PaperStructureTree`` whose leaf
  nodes carry their own page-cited text.
- ``extract_knowledge`` returns a page-aware chunk set and one cited global
  summary (``PaperFlowResult``).

Every method is a pure ``-> artifact`` derivation of the immutable source; no
method mutates it and nothing accumulates between calls. The handle binds no
library, persists nothing, and retrieves nothing — persistence (``library``)
and retrieval (``mind``) are downstream concerns a caller wires itself.

The module keeps only what genuinely needs both the preprocess/rag value
objects and IO: fetching bytes, parsing the PDF, reading page-asset bytes off
disk, and mapping those ephemeral, path-based artifacts into knowledge-native
inputs. All identity (IDs, content and producer hashes, citation resolution)
lives on the knowledge models' ``from_*`` constructors, so this module imports
no private ID helpers and computes no paper ID itself.

``paper_flow`` remains as a thin compatibility function delegating to
``PaperFlow.open(...).extract_knowledge(...)``.
"""

import mimetypes
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from quantmind.configs import PaperFlowCfg, PaperStructureCfg
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
from quantmind.flows.paper._structure import (
    PaperStructureError,
    _AgentsPaperStructureProvider,
    _PaperStructureProvider,
    _structure_instructions_hash,
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
    PaperStructureProducer,
    PaperStructureTree,
    PaperSummaryProducer,
)
from quantmind.preprocess import (
    BoundingBox,
    ParsedDocument,
    ParsedPage,
    TextBlock,
    extract_outline_signals,
)
from quantmind.preprocess.fetch import (
    Fetched,
    RawPaper,
    fetch_arxiv,
    fetch_url,
    read_local_file,
)
from quantmind.preprocess.format import parse_pdf
from quantmind.rag import (
    ParsedChunk,
    SentenceSplitterConfig,
    chunk_parsed_document,
)

__all__ = [
    "PaperFlow",
    "PaperStructureError",
    "UnsupportedContentTypeError",
    "paper_flow",
]


class UnsupportedContentTypeError(ValueError):
    """The source is not a page-aware PDF supported by Paper Flow V1."""


class PaperFlow:
    """Document-scoped handle grouping the finished paper pipelines.

    ``open`` performs the one expensive step the pipelines share — fetch and
    parse — exactly once and binds the resulting immutable source revision (and
    the parsed document that chunking needs). Every pipeline method is a pure
    ``-> artifact`` derivation of that immutable state; the handle binds no
    library, persists nothing, and retrieves nothing.
    """

    __slots__ = (
        "_parsed",
        "_source",
        "_structure_provider",
        "_summary_provider",
    )

    def __init__(
        self,
        *,
        source: PaperSourceRevision,
        parsed: ParsedDocument,
        structure_provider: _PaperStructureProvider | None = None,
        summary_provider: _PaperSummaryProvider | None = None,
    ) -> None:
        self._source = source
        self._parsed = parsed
        self._structure_provider = structure_provider
        self._summary_provider = summary_provider

    @classmethod
    async def open(
        cls,
        input: PaperInput,
        *,
        output_dir: str | None = None,
        _structure_provider: _PaperStructureProvider | None = None,
        _summary_provider: _PaperSummaryProvider | None = None,
    ) -> "PaperFlow":
        """Fetch and parse one source exactly once, returning a bound handle.

        The parsed document and the exact ``PaperSourceRevision`` are stored as
        immutable instance state; no later method re-fetches or re-parses.

        Args:
            input: Typed paper source. V1 requires a PDF-backed input.
            output_dir: Directory for parser-written page assets, forwarded to
                ``parse_pdf``.
            _structure_provider: Optional test seam for the structure draft.
            _summary_provider: Optional test seam for the summary draft.

        Returns:
            A handle bound to the immutable parsed source.

        Raises:
            UnsupportedContentTypeError: If the resolved content is not a PDF.
            NotImplementedError: If a DOI input has no exact open PDF resolver.
        """
        facts = await _fetch_paper_source(input)
        parsed = await parse_pdf(facts.raw_bytes, artifact_dir=output_dir)
        source = PaperSourceRevision.from_parsed(
            facts=facts,
            source_hash=parsed.source_hash,
            parser_name=parsed.parser_name,
            parser_version=parsed.parser_version,
            cleanup_version=parsed.cleanup_version,
            pages=_adapt_pages(parsed),
        )
        return cls(
            source=source,
            parsed=parsed,
            structure_provider=_structure_provider,
            summary_provider=_summary_provider,
        )

    @property
    def source(self) -> PaperSourceRevision:
        """The immutable source revision fetched and parsed at ``open``."""
        return self._source

    async def build_structure(
        self,
        *,
        cfg: PaperStructureCfg | None = None,
    ) -> PaperStructureTree:
        """Build one self-contained page-preserving structure tree.

        Deterministic outline signals seed a single draft-structuring agent;
        code then mints identity, resolves page citations, and — inside the
        knowledge-layer constructor — populates each leaf node's ``content``
        from its cited source pages. The returned tree is self-contained: it
        yields node text without the source revision or a chunk set.

        Args:
            cfg: Model, prompt, input, and tree bounds. A default
                ``PaperStructureCfg`` is used when omitted.

        Returns:
            A code-identified tree whose leaf nodes carry cited page text.

        Raises:
            PaperStructureError: If the model call exceeds its timeout.
            StructureTreeValidationError: If the proposed page tree is invalid.
        """
        cfg = cfg or PaperStructureCfg()
        provider = self._structure_provider or _AgentsPaperStructureProvider()
        signals = extract_outline_signals(_parsed_document(self._source))
        draft = await provider.structure(signals, self._source, cfg=cfg)
        producer = PaperStructureProducer(
            model=cfg.model,
            prompt_version=cfg.prompt_version,
            instructions_hash=_structure_instructions_hash(cfg),
            page_text_chars=cfg.page_text_chars,
            max_output_tokens=cfg.max_output_tokens,
            max_depth=cfg.max_depth,
            max_nodes=cfg.max_nodes,
        )
        return PaperStructureTree.from_draft(
            self._source,
            producer=producer,
            draft=draft,
        )

    async def extract_knowledge(
        self,
        *,
        cfg: PaperFlowCfg | None = None,
    ) -> PaperFlowResult:
        """Build a page-aware chunk set and one cited global summary.

        IDs, source metadata, artifact membership, lineage, and citation links
        are minted and validated by the knowledge-layer constructors. The model
        returns only summary prose and chunk/page coordinates through a bounded
        seam.

        Args:
            cfg: Splitter, summary model, and usage/runtime limits. A default
                ``PaperFlowCfg`` is used when omitted.

        Returns:
            The exact source revision, one chunk-set artifact, and one cited
            global-summary artifact.

        Raises:
            PaperCitationValidationError: If generated citations are invalid or
                do not meet the configured source-coverage policy.
        """
        cfg = cfg or PaperFlowCfg()
        parsed_chunks = chunk_parsed_document(
            self._parsed,
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
            self._source,
            _adapt_chunks(parsed_chunks, self._source),
            producer=producer,
        )
        provider = self._summary_provider or _AgentsPaperSummaryProvider()
        draft = await provider.summarize(self._source, chunk_set, cfg=cfg)
        summary = _build_summary(chunk_set, draft, cfg)
        return PaperFlowResult(
            source_revision=self._source,
            chunk_set=chunk_set,
            global_summary=summary,
        )


async def paper_flow(
    input: PaperInput,
    *,
    cfg: PaperFlowCfg | None = None,
    _summary_provider: _PaperSummaryProvider | None = None,
) -> PaperFlowResult:
    """Build a page-aware chunk set and one cited global summary.

    Thin compatibility wrapper over the ``PaperFlow`` handle: it opens the
    document (fetch + parse once) and runs the knowledge-extraction pipeline.
    New code should call ``PaperFlow.open(...).extract_knowledge(...)`` directly.

    Args:
        input: Typed paper source. V1 requires a PDF-backed input.
        cfg: Splitter, summary model, and usage/runtime limits.
        _summary_provider: Optional test seam for the summary draft.

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
    paper = await PaperFlow.open(
        input,
        output_dir=cfg.output_dir,
        _summary_provider=_summary_provider,
    )
    return await paper.extract_knowledge(cfg=cfg)


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


def _parsed_document(source: PaperSourceRevision) -> ParsedDocument:
    """Project a canonical source manifest into outline extraction input."""
    return ParsedDocument(
        source_hash=source.parsed.source_hash,
        parser_name=source.parsed.parser_name,
        parser_version=source.parsed.parser_version,
        cleanup_version=source.parsed.cleanup_version,
        pages=tuple(
            ParsedPage(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                text=page.text,
                blocks=tuple(
                    TextBlock(
                        text=block.text,
                        page_number=page.page_number,
                        bbox=BoundingBox(
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
            )
            for page in source.parsed.pages
        ),
    )
