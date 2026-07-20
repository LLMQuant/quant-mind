"""Tests for page-aware PDF parsing."""

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from quantmind.preprocess.format.pdf import (
    PdfParseError,
    parse_pdf,
    pdf_to_markdown,
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
