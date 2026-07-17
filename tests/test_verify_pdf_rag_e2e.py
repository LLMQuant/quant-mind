import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from quantmind.preprocess import (
    ParsedChunk,
    ParsedDocument,
    ParsedDocumentHit,
    ParsedPage,
)
from scripts import verify_pdf_rag_e2e


def _document(page_count: int = 15) -> ParsedDocument:
    return ParsedDocument(
        source_hash="hash",
        parser_name="liteparse",
        parser_version="2.6.0",
        cleanup_version="1",
        pages=tuple(
            ParsedPage(
                page_number=page,
                width=612,
                height=792,
                text=f"page {page}",
                blocks=(),
            )
            for page in range(1, page_count + 1)
        ),
    )


def _chunk(text: str, page: int = 5) -> ParsedChunk:
    return ParsedChunk(
        chunk_id=f"chunk-{page}",
        text=text,
        source_hash="hash",
        page_number=page,
        block_boxes=(),
        screenshot_path=None,
        image_paths=(),
    )


class VerifyPdfRagE2ETests(unittest.IsolatedAsyncioTestCase):
    async def test_main_passes_pinned_page_and_relevance_checks(self):
        paper = SimpleNamespace(
            arxiv_id="1706.03762v7",
            bytes=b"pdf",
        )
        document = _document()
        chunks = (_chunk("Multi-head attention projects queries."),)
        hits = (ParsedDocumentHit(chunk=chunks[0], score=1.0),)
        with (
            patch.object(
                verify_pdf_rag_e2e,
                "fetch_arxiv",
                new=AsyncMock(return_value=paper),
            ),
            patch.object(
                verify_pdf_rag_e2e,
                "parse_pdf",
                new=AsyncMock(return_value=document),
            ),
            patch.object(
                verify_pdf_rag_e2e,
                "chunk_parsed_document",
                return_value=chunks,
            ),
            patch.object(
                verify_pdf_rag_e2e,
                "retrieve_parsed_document",
                return_value=hits,
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            exit_code = await verify_pdf_rag_e2e.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("[PASS] pdf-rag", output.getvalue())
        self.assertIn("top_pages=[5]", output.getvalue())

    async def test_main_reports_upstream_failure(self):
        with (
            patch.object(
                verify_pdf_rag_e2e,
                "fetch_arxiv",
                new=AsyncMock(side_effect=TimeoutError("bounded timeout")),
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            exit_code = await verify_pdf_rag_e2e.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("[FAIL] pdf-rag: TimeoutError", output.getvalue())

    async def test_main_rejects_wrong_page_count_or_irrelevant_hits(self):
        paper = SimpleNamespace(
            arxiv_id="1706.03762v7",
            bytes=b"pdf",
        )
        document = _document(page_count=14)
        chunks = (_chunk("Unrelated passage"),)
        hits = (ParsedDocumentHit(chunk=chunks[0], score=1.0),)
        with (
            patch.object(
                verify_pdf_rag_e2e,
                "fetch_arxiv",
                new=AsyncMock(return_value=paper),
            ),
            patch.object(
                verify_pdf_rag_e2e,
                "parse_pdf",
                new=AsyncMock(return_value=document),
            ),
            patch.object(
                verify_pdf_rag_e2e,
                "chunk_parsed_document",
                return_value=chunks,
            ),
            patch.object(
                verify_pdf_rag_e2e,
                "retrieve_parsed_document",
                return_value=hits,
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            exit_code = await verify_pdf_rag_e2e.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("[FAIL] pdf-rag", output.getvalue())


if __name__ == "__main__":
    unittest.main()
