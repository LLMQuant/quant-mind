#!/usr/bin/env python3
"""Run the bounded live Transformer PDF parsing and retrieval smoke test."""

import asyncio

from quantmind.preprocess import (
    SentenceSplitterConfig,
    chunk_parsed_document,
    fetch_arxiv,
    parse_pdf,
    retrieve_parsed_document,
)

_ARXIV_ID = "1706.03762v7"


async def main() -> int:
    """Fetch the pinned paper and verify page-aware retrieval."""
    try:
        paper = await asyncio.wait_for(fetch_arxiv(_ARXIV_ID), timeout=120)
        document = await asyncio.wait_for(parse_pdf(paper.bytes), timeout=120)
        chunks = chunk_parsed_document(
            document,
            config=SentenceSplitterConfig(chunk_size=512, chunk_overlap=64),
        )
        hits = retrieve_parsed_document(
            chunks,
            "How does multi-head attention work?",
            top_k=5,
        )
    except Exception as exc:
        print(f"[FAIL] pdf-rag: {type(exc).__name__}: {exc}")
        return 1

    relevant = [
        hit for hit in hits if "multi-head attention" in hit.chunk.text.lower()
    ]
    passed = (
        paper.arxiv_id == _ARXIV_ID
        and len(document.pages) == 15
        and bool(chunks)
        and bool(relevant)
        and all(hit.chunk.page_number >= 1 for hit in hits)
    )
    state = "PASS" if passed else "FAIL"
    print(
        f"[{state}] pdf-rag: arxiv={paper.arxiv_id} "
        f"pages={len(document.pages)} chunks={len(chunks)} "
        f"top_pages={[hit.chunk.page_number for hit in hits]}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
