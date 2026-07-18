"""Parse a local PDF and retrieve page-aware evidence."""

import asyncio
from pathlib import Path

from quantmind.preprocess.format import parse_pdf
from quantmind.rag import chunk_parsed_document, retrieve_parsed_document


async def main() -> None:
    """Parse one PDF and print page-aware BM25 evidence."""
    pdf_path = Path("tests/fixtures/paper/golden/paper.pdf")
    document = await parse_pdf(pdf_path.read_bytes())
    chunks = chunk_parsed_document(document)
    hits = retrieve_parsed_document(
        chunks,
        "How is the long-short portfolio constructed?",
        top_k=3,
    )
    for hit in hits:
        print(
            f"page={hit.chunk.page_number} score={hit.score:.3f} "
            f"text={hit.chunk.text[:120]!r}"
        )


if __name__ == "__main__":
    asyncio.run(main())
