"""Tests for page-aware PDF parsing and LlamaIndex ingestion."""

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from quantmind.preprocess.format.pdf import (
    PdfParseError,
    SentenceSplitterConfig,
    chunk_parsed_document,
    parse_pdf,
    pdf_to_markdown,
    retrieve_parsed_document,
)

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "tiny.pdf"
_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "paper"
    / "golden"
    / "paper.pdf"
)


class PdfToMarkdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_extracts_fixture_text(self):
        pdf_bytes = _FIXTURE.read_bytes()
        result = await pdf_to_markdown(pdf_bytes)
        self.assertIn("QuantMind Test Fixture", result)
        self.assertIn("plain text", result)

    async def test_empty_input_raises(self):
        with self.assertRaises(PdfParseError):
            await pdf_to_markdown(b"")

    async def test_invalid_bytes_raises(self):
        with self.assertRaises(PdfParseError):
            await pdf_to_markdown(b"this is not a pdf at all")

    async def test_returns_str(self):
        pdf_bytes = _FIXTURE.read_bytes()
        result = await pdf_to_markdown(pdf_bytes)
        self.assertIsInstance(result, str)

    async def test_golden_preserves_pages_blocks_coordinates_and_artifacts(
        self,
    ):
        pdf_bytes = _GOLDEN.read_bytes()
        with TemporaryDirectory() as artifact_dir:
            document = await parse_pdf(pdf_bytes, artifact_dir=artifact_dir)

            self.assertEqual(
                document.source_hash, hashlib.sha256(pdf_bytes).hexdigest()
            )
            self.assertEqual(document.parser_name, "liteparse")
            self.assertEqual(
                [page.page_number for page in document.pages], [1, 2, 3, 4]
            )
            self.assertIn(
                "A Synthetic Cross-Sectional Momentum Study",
                document.pages[0].text,
            )
            self.assertTrue(all(page.blocks for page in document.pages))
            self.assertTrue(
                all(
                    block.page_number == page.page_number
                    for page in document.pages
                    for block in page.blocks
                )
            )
            self.assertTrue(
                all(
                    block.bbox.x1 >= block.bbox.x0
                    and block.bbox.y1 >= block.bbox.y0
                    for page in document.pages
                    for block in page.blocks
                )
            )
            self.assertTrue(
                all(
                    page.screenshot_path is not None
                    and Path(page.screenshot_path).is_file()
                    for page in document.pages
                )
            )

    async def test_llamaindex_chunks_and_bm25_hits_keep_page_evidence(self):
        document = await parse_pdf(_GOLDEN.read_bytes())
        chunks = chunk_parsed_document(
            document,
            config=SentenceSplitterConfig(chunk_size=256, chunk_overlap=32),
        )

        self.assertTrue(chunks)
        self.assertEqual({chunk.page_number for chunk in chunks}, {1, 2, 3, 4})
        self.assertTrue(
            all(chunk.source_hash == document.source_hash for chunk in chunks)
        )
        self.assertTrue(all(chunk.block_boxes for chunk in chunks))

        hits = retrieve_parsed_document(
            chunks,
            "equal-weighted quintiles long-short portfolio",
            top_k=2,
        )
        self.assertEqual(len(hits), 2)
        self.assertIn(hits[0].chunk.page_number, {3, 4})
        self.assertEqual(hits[0].chunk.source_hash, document.source_hash)

    async def test_empty_physical_page_is_not_dropped_or_renumbered(self):
        source = pymupdf.open()
        try:
            source.new_page().insert_text((72, 72), "first page")
            source.new_page()
            source.new_page().insert_text((72, 72), "third page")
            document = await parse_pdf(source.tobytes())
        finally:
            source.close()

        self.assertEqual(
            [page.page_number for page in document.pages], [1, 2, 3]
        )
        self.assertEqual(document.pages[1].text, "")
        self.assertEqual(document.pages[1].blocks, ())
        self.assertIn("third page", document.pages[2].text)

    async def test_retrieval_rejects_invalid_query_arguments(self):
        document = await parse_pdf(_FIXTURE.read_bytes())
        chunks = chunk_parsed_document(document)
        with self.assertRaisesRegex(ValueError, "query"):
            retrieve_parsed_document(chunks, "   ")
        with self.assertRaisesRegex(ValueError, "top_k"):
            retrieve_parsed_document(chunks, "fixture", top_k=0)
