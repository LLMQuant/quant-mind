"""Build, persist, reopen, and search the Transformer paper artifacts."""

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import ArxivIdentifier
from quantmind.flows import paper_flow
from quantmind.knowledge import PaperChunk, PaperGlobalSummary
from quantmind.library import (
    LocalKnowledgeLibrary,
    SemanticHit,
    SemanticQuery,
)

_ARXIV_ID = "1706.03762v7"
_EMBEDDING_MODEL = "text-embedding-3-small"


async def _search_and_resolve(
    library: LocalKnowledgeLibrary,
) -> tuple[
    list[SemanticHit],
    list[SemanticHit],
    Sequence[object],
]:
    """Run both V1 retrieval grains and resolve every returned locator."""
    summary_hits = await library.search(
        SemanticQuery(
            text="What is the paper's central contribution?",
            artifact_kinds=["paper_summary"],
            top_k=3,
        )
    )
    chunk_hits = await library.search(
        SemanticQuery(
            text="How does multi-head attention work?",
            artifact_kinds=["paper_chunk_set"],
            top_k=5,
        )
    )
    resolved = [
        await library.resolve(hit.locator)
        for hit in (*summary_hits, *chunk_hits)
    ]
    return summary_hits, chunk_hits, resolved


async def main() -> None:
    """Run the common Paper Flow V1 path and print auditable evidence."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    workspace = Path(".quantmind")
    workspace.mkdir(exist_ok=True)
    result = await paper_flow(
        ArxivIdentifier(id=_ARXIV_ID),
        cfg=PaperFlowCfg(
            model="gpt-4o-mini",
            output_dir=str(workspace / "attention-assets"),
        ),
    )

    database = workspace / "library.db"
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

    library = await LocalKnowledgeLibrary.open(
        database,
        embedding_model=_EMBEDDING_MODEL,
    )
    try:
        restored = await library.get_paper(
            result.source_revision.id,
            chunk_set_id=result.chunk_set.id,
            summary_id=result.global_summary.id,
        )
        (
            second_summary_hits,
            second_chunk_hits,
            second_resolved,
        ) = await _search_and_resolve(library)

        print(restored.global_summary.summary)
        print(
            f"chunks={len(restored.chunk_set.chunks)} "
            f"source_pages={len(restored.source_revision.parsed.pages)}"
        )
        print(
            "citations="
            + ", ".join(
                f"page {citation.page_number} / chunk {citation.chunk_id}"
                for citation in restored.global_summary.citations
            )
        )
        print(
            "scores_before_reopen="
            f"summary={[hit.score for hit in first_summary_hits]} "
            f"chunks={[hit.score for hit in first_chunk_hits]}"
        )
        print(
            "scores_after_reopen="
            f"summary={[hit.score for hit in second_summary_hits]} "
            f"chunks={[hit.score for hit in second_chunk_hits]}"
        )
        for hit, resolved in zip(
            (*second_summary_hits, *second_chunk_hits),
            second_resolved,
            strict=True,
        ):
            if isinstance(resolved, PaperGlobalSummary):
                detail = "global summary"
            elif isinstance(resolved, PaperChunk):
                pages = sorted(
                    {span.page_number for span in resolved.source_spans}
                )
                detail = f"chunk pages={pages} text={resolved.text[:120]!r}"
            else:
                detail = type(resolved).__name__
            print(f"score={hit.score:.3f} {detail}")
        print(
            f"resolved_before_reopen={len(first_resolved)} "
            f"resolved_after_reopen={len(second_resolved)}"
        )
    finally:
        await library.close()


if __name__ == "__main__":
    asyncio.run(main())
