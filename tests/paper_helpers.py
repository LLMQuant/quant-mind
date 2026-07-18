"""Deterministic paper artifacts shared by offline tests."""

import hashlib
from datetime import datetime, timezone

from quantmind.knowledge import (
    ArtifactLocator,
    PaperAssetRef,
    PaperChunk,
    PaperChunkingConfig,
    PaperChunkSet,
    PaperCitation,
    PaperFlowResult,
    PaperGlobalSummary,
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

_RAW_BYTES = b"deterministic paper source revision"
_WHEN = datetime(2017, 12, 6, tzinfo=timezone.utc)


def build_paper_result(
    *,
    chunk_size: int = 128,
    summary_model: str = "fake-summary",
    summary_text: str = (
        "The Transformer replaces recurrence and convolution with an "
        "encoder-decoder attention architecture, uses multi-head attention, "
        "and improves translation quality with efficient training."
    ),
) -> PaperFlowResult:
    """Build one valid two-page result without parsing, network, or models."""
    source_hash = hashlib.sha256(_RAW_BYTES).hexdigest()
    source_id = _paper_source_id(source_hash)
    raw_asset = PaperAssetRef(
        asset_id=_paper_asset_id(
            source_id,
            kind="raw",
            page_number=None,
            content_hash=source_hash,
        ),
        kind="raw",
        media_type="application/pdf",
        content_hash=source_hash,
        size_bytes=len(_RAW_BYTES),
    )
    source = PaperSourceRevision(
        id=source_id,
        source=SourceRef(
            kind="arxiv",
            uri="https://arxiv.org/pdf/1706.03762v7.pdf",
            fetched_at=_WHEN,
            content_hash=source_hash,
        ),
        as_of=_WHEN,
        available_at=_WHEN,
        published_at=_WHEN,
        arxiv_id="1706.03762v7",
        title="Attention Is All You Need",
        authors=("Ashish Vaswani",),
        parsed=PaperParsedManifest(
            source_hash=source_hash,
            parser_name="fake-parser",
            parser_version="1",
            cleanup_version="1",
            pages=(
                PaperParsedPage(
                    page_number=1,
                    width=612,
                    height=792,
                    text=(
                        "The Transformer removes recurrence and convolution "
                        "from sequence transduction."
                    ),
                ),
                PaperParsedPage(
                    page_number=2,
                    width=612,
                    height=792,
                    text=(
                        "Multi-head attention uses parallel projections. "
                        "The model improves translation and training efficiency."
                    ),
                ),
            ),
        ),
        raw_asset_id=raw_asset.asset_id,
        assets=(raw_asset,),
        blobs={source_hash: _RAW_BYTES},
    )

    producer = PaperChunkingConfig(
        splitter_version="fake-llama-index",
        chunk_size=chunk_size,
        chunk_overlap=min(16, chunk_size - 1),
    )
    producer_hash = _stable_hash(producer.model_dump(mode="json"))
    chunk_set_id = _paper_artifact_id(
        source_id,
        "paper_chunk_set",
        producer_hash,
    )
    chunk_values = (
        (
            1,
            "The Transformer removes recurrence and convolution.",
        ),
        (2, "Multi-head attention uses parallel learned projections."),
        (2, "The model improves translation and training efficiency."),
    )
    chunks: list[PaperChunk] = []
    for position, (page, text) in enumerate(chunk_values):
        span = PaperSourceSpan(
            page_number=page,
            start_char=position * 10,
            end_char=position * 10 + len(text),
        )
        text_hash = _text_hash(text)
        chunks.append(
            PaperChunk(
                chunk_id=_paper_chunk_id(
                    chunk_set_id,
                    position=position,
                    content_hash=text_hash,
                    spans=(span,),
                ),
                chunk_set_id=chunk_set_id,
                source_revision_id=source_id,
                position=position,
                text=text,
                content_hash=text_hash,
                source_spans=(span,),
            )
        )
    chunk_tuple = tuple(chunks)
    chunk_set = PaperChunkSet(
        id=chunk_set_id,
        source_revision_id=source_id,
        producer=producer,
        producer_config_hash=producer_hash,
        content_hash=_paper_chunk_set_content_hash(chunk_tuple),
        chunks=chunk_tuple,
    )

    summary_producer = PaperSummaryProducer(
        model=summary_model,
        prompt_version="test-v1",
        input_chunk_set_id=chunk_set_id,
        instructions_hash=hashlib.sha256(b"test instructions").hexdigest(),
        max_output_tokens=512,
    )
    summary_config_hash = _stable_hash(summary_producer.model_dump(mode="json"))
    citations = tuple(
        PaperCitation(
            chunk_set_id=chunk_set_id,
            chunk_id=chunk.chunk_id,
            page_number=chunk.source_spans[0].page_number,
        )
        for chunk in chunk_tuple
    )
    summary = PaperGlobalSummary(
        id=_paper_artifact_id(
            source_id,
            "paper_summary",
            summary_config_hash,
        ),
        source_revision_id=source_id,
        producer=summary_producer,
        producer_config_hash=summary_config_hash,
        content_hash=_paper_summary_content_hash(summary_text, citations),
        summary=summary_text,
        citations=citations,
        derived_from=(
            ArtifactLocator(
                source_revision_id=source_id,
                artifact_id=chunk_set_id,
                artifact_kind="paper_chunk_set",
            ),
        ),
    )
    return PaperFlowResult(
        source_revision=source,
        chunk_set=chunk_set,
        global_summary=summary,
    )
