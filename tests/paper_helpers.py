"""Deterministic paper artifacts shared by offline tests.

The fixture builds through the same knowledge-layer smart constructors that
``paper_flow`` uses, so tests exercise the real identity path instead of
re-deriving IDs and hashes with the private helpers.
"""

import hashlib
from datetime import datetime, timezone

from quantmind.knowledge import (
    PaperChunkingConfig,
    PaperChunkInput,
    PaperChunkSet,
    PaperCitationDraft,
    PaperFlowResult,
    PaperGlobalSummary,
    PaperPageInput,
    PaperSourceFacts,
    PaperSourceRevision,
    PaperSummaryProducer,
)

_RAW_BYTES = b"deterministic paper source revision"
_WHEN = datetime(2017, 12, 6, tzinfo=timezone.utc)
_PAGE_ONE = (
    "The Transformer removes recurrence and convolution "
    "from sequence transduction."
)
_PAGE_TWO = (
    "Multi-head attention uses parallel projections. "
    "The model improves translation and training efficiency."
)


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
    source = PaperSourceRevision.from_parsed(
        facts=PaperSourceFacts(
            kind="arxiv",
            uri="https://arxiv.org/pdf/1706.03762v7.pdf",
            media_type="application/pdf",
            raw_bytes=_RAW_BYTES,
            fetched_at=_WHEN,
            available_at=_WHEN,
            published_at=_WHEN,
            arxiv_id="1706.03762v7",
            title="Attention Is All You Need",
            authors=("Ashish Vaswani",),
        ),
        source_hash=source_hash,
        parser_name="fake-parser",
        parser_version="1",
        cleanup_version="1",
        pages=(
            PaperPageInput(
                page_number=1, width=612, height=792, text=_PAGE_ONE
            ),
            PaperPageInput(
                page_number=2, width=612, height=792, text=_PAGE_TWO
            ),
        ),
    )

    chunk_values = (
        (1, "The Transformer removes recurrence and convolution."),
        (2, "Multi-head attention uses parallel learned projections."),
        (2, "The model improves translation and training efficiency."),
    )
    chunk_inputs = tuple(
        PaperChunkInput(
            page_number=page,
            start_char=position * 10,
            end_char=position * 10 + len(text),
            block_boxes=(),
            text=text,
        )
        for position, (page, text) in enumerate(chunk_values)
    )
    chunk_set = PaperChunkSet.from_parsed_chunks(
        source,
        chunk_inputs,
        producer=PaperChunkingConfig(
            splitter_version="fake-llama-index",
            chunk_size=chunk_size,
            chunk_overlap=min(16, chunk_size - 1),
        ),
    )

    summary = PaperGlobalSummary.from_draft(
        chunk_set,
        producer=PaperSummaryProducer(
            model=summary_model,
            prompt_version="test-v1",
            input_chunk_set_id=chunk_set.id,
            instructions_hash=hashlib.sha256(b"test instructions").hexdigest(),
            max_output_tokens=512,
            research_group_size=8,
        ),
        summary=summary_text,
        citations=tuple(
            PaperCitationDraft(
                chunk_index=index,
                page_number=chunk.source_spans[0].page_number,
            )
            for index, chunk in enumerate(chunk_set.chunks)
        ),
        min_citations=1,
        min_pages=1,
    )
    return PaperFlowResult(
        source_revision=source,
        chunk_set=chunk_set,
        global_summary=summary,
    )
