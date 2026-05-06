"""Tests for preprocess.format.pdf — pdf_to_markdown via PyMuPDF."""

import unittest
from pathlib import Path

from quantmind.preprocess.format.pdf import PdfParseError, pdf_to_markdown

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "tiny.pdf"


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
