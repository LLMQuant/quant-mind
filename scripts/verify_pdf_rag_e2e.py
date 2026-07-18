#!/usr/bin/env python3
"""Run the bounded live *Attention Is All You Need* Paper Flow V1 slice."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import ArxivIdentifier
from quantmind.flows import paper_flow
from quantmind.knowledge import (
    PaperArtifactKind,
    PaperChunk,
    PaperGlobalSummary,
)
from quantmind.library import LocalKnowledgeLibrary, SemanticQuery

_ARXIV_ID = "1706.03762v7"
_EMBEDDING_MODEL = "text-embedding-3-small"
_SUMMARY_QUERY = "What is the paper's central contribution?"
_CHUNK_QUERY = "How does multi-head attention work?"


async def _search_and_resolve(
    library: LocalKnowledgeLibrary,
) -> tuple[list[Any], list[Any], list[Any]]:
    summary_hits = await library.search(
        SemanticQuery(
            text=_SUMMARY_QUERY,
            artifact_kinds=[PaperArtifactKind.GLOBAL_SUMMARY],
            top_k=3,
        )
    )
    chunk_hits = await library.search(
        SemanticQuery(
            text=_CHUNK_QUERY,
            artifact_kinds=[PaperArtifactKind.CHUNK_SET],
            top_k=5,
        )
    )
    resolved = [
        await library.resolve(hit.locator)
        for hit in (*summary_hits, *chunk_hits)
    ]
    return summary_hits, chunk_hits, resolved


async def _run_vertical_slice() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="quantmind-paper-v1-") as directory:
        root = Path(directory)
        result = await paper_flow(
            ArxivIdentifier(id=_ARXIV_ID),
            cfg=PaperFlowCfg(
                model="gpt-4o-mini",
                output_dir=str(root / "assets"),
                timeout_seconds=240,
                summary_research_group_size=8,
                summary_concurrency=2,
                max_summary_output_tokens=4_096,
                min_summary_citations=3,
                min_summary_pages=2,
            ),
        )
        database = root / "library.db"
        library = await LocalKnowledgeLibrary.open(
            database,
            embedding_model=_EMBEDDING_MODEL,
        )
        try:
            await library.put_paper(result)
            (
                first_summary_hits,
                first_chunk_hits,
                first_resolved,
            ) = await _search_and_resolve(library)
        finally:
            await library.close()

        reopened = await LocalKnowledgeLibrary.open(
            database,
            embedding_model=_EMBEDDING_MODEL,
        )
        try:
            restored = await reopened.get_paper(
                result.source_revision.id,
                chunk_set_id=result.chunk_set.id,
                summary_id=result.global_summary.id,
            )
            (
                second_summary_hits,
                second_chunk_hits,
                second_resolved,
            ) = await _search_and_resolve(reopened)
        finally:
            await reopened.close()

    return {
        "arxiv_id": result.source_revision.arxiv_id,
        "page_count": len(result.source_revision.parsed.pages),
        "text_page_count": sum(
            bool(page.text.strip())
            for page in result.source_revision.parsed.pages
        ),
        "asset_count": len(result.source_revision.assets),
        "screenshot_pages": [
            page.page_number
            for page in result.source_revision.parsed.pages
            if page.screenshot_asset_id is not None
        ],
        "chunk_asset_reference_count": sum(
            len(span.asset_ids)
            for chunk in result.chunk_set.chunks
            for span in chunk.source_spans
        ),
        "chunk_count": len(result.chunk_set.chunks),
        "summary": result.global_summary.summary,
        "summary_orchestration": (result.global_summary.producer.orchestration),
        "citation_count": len(result.global_summary.citations),
        "citation_pages": sorted(
            {
                citation.page_number
                for citation in result.global_summary.citations
            }
        ),
        "first_summary_scores": [hit.score for hit in first_summary_hits],
        "first_chunk_scores": [hit.score for hit in first_chunk_hits],
        "second_summary_scores": [hit.score for hit in second_summary_hits],
        "second_chunk_scores": [hit.score for hit in second_chunk_hits],
        "first_summary_resolved": any(
            isinstance(value, PaperGlobalSummary) for value in first_resolved
        ),
        "second_summary_resolved": any(
            isinstance(value, PaperGlobalSummary) for value in second_resolved
        ),
        "first_multi_head_pages": [
            value.source_spans[0].page_number
            for value in first_resolved
            if isinstance(value, PaperChunk)
            and "multi-head attention" in value.text.lower()
        ],
        "second_multi_head_pages": [
            value.source_spans[0].page_number
            for value in second_resolved
            if isinstance(value, PaperChunk)
            and "multi-head attention" in value.text.lower()
        ],
        "restored": restored.chunk_set.id == result.chunk_set.id
        and restored.global_summary.id == result.global_summary.id,
        "embedding_model": _EMBEDDING_MODEL,
    }


def _summary_has_required_coverage(summary: str) -> bool:
    text = summary.lower()
    attention_only = "attention" in text and any(
        marker in text
        for marker in (
            "attention-only",
            "attention based",
            "attention-based",
            "solely",
            "entirely",
            "exclusively",
            "purely",
            "eliminat",
            "without recurrence",
            "remov",
            "replac",
            "eschew",
        )
    )
    return (
        all(
            any(term in text for term in group)
            for group in (
                ("recurrence", "recurrent"),
                ("convolution", "convolutional"),
                (
                    "encoder-decoder",
                    "encoder",
                    "decoder",
                    "multi-head attention",
                    "multihead attention",
                ),
                ("translation", "training efficiency", "training time"),
            )
        )
        and attention_only
    )


def _passed(snapshot: dict[str, Any]) -> bool:
    return bool(
        snapshot["arxiv_id"] == _ARXIV_ID
        and snapshot["page_count"] == 15
        and snapshot["text_page_count"] == 15
        and snapshot["asset_count"] > 1
        and len(snapshot["screenshot_pages"]) == 15
        and snapshot["chunk_asset_reference_count"] > 0
        and snapshot["chunk_count"] > 0
        and snapshot["summary"]
        and snapshot["summary_orchestration"] == "map-reduce-v1"
        and snapshot["citation_count"] >= 3
        and len(snapshot["citation_pages"]) >= 2
        and snapshot["first_summary_scores"]
        and snapshot["second_summary_scores"]
        and snapshot["first_multi_head_pages"]
        and snapshot["second_multi_head_pages"]
        and snapshot["first_summary_resolved"]
        and snapshot["second_summary_resolved"]
        and snapshot["restored"]
        and snapshot["embedding_model"] == _EMBEDDING_MODEL
        and _summary_has_required_coverage(snapshot["summary"])
    )


async def main() -> int:
    """Run and report the timeout-bounded live vertical slice."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("[FAIL] paper-flow-v1: OPENAI_API_KEY is not set")
        return 1
    try:
        snapshot = await asyncio.wait_for(_run_vertical_slice(), timeout=480)
    except Exception as exc:
        print(f"[FAIL] paper-flow-v1: {type(exc).__name__}: {exc}")
        return 1

    state = "PASS" if _passed(snapshot) else "FAIL"
    print(
        f"[{state}] paper-flow-v1: arxiv={snapshot['arxiv_id']} "
        f"pages={snapshot['page_count']} "
        f"text_pages={snapshot['text_page_count']} "
        f"chunks={snapshot['chunk_count']} assets={snapshot['asset_count']} "
        f"chunk_asset_refs={snapshot['chunk_asset_reference_count']} "
        f"orchestration={snapshot['summary_orchestration']} "
        f"asset_pages={snapshot['screenshot_pages']} "
        f"citation_pages={snapshot['citation_pages']} "
        f"summary_scores={snapshot['second_summary_scores']} "
        f"chunk_scores={snapshot['second_chunk_scores']} "
        f"multi_head_pages={snapshot['second_multi_head_pages']}"
    )
    print(f"Summary:\n{snapshot['summary']}")
    return 0 if state == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
