"""Tests for preprocess.fetch.local — read_local_file."""

import tempfile
import unittest
from pathlib import Path

from quantmind.preprocess.fetch.local import read_local_file


class ReadLocalFileTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_pdf_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.pdf"
            path.write_bytes(b"%PDF-1.4\nfake\n")
            result = await read_local_file(path)
        self.assertEqual(result.bytes, b"%PDF-1.4\nfake\n")
        self.assertEqual(result.content_type, "application/pdf")
        self.assertTrue(result.source_url.startswith("file://"))
        self.assertEqual(result.headers, {})

    async def test_html_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.html"
            path.write_bytes(b"<html></html>")
            result = await read_local_file(path)
        self.assertEqual(result.content_type, "text/html")

    async def test_markdown_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.md"
            path.write_bytes(b"# title")
            result = await read_local_file(path)
        self.assertEqual(result.content_type, "text/markdown")

    async def test_unknown_suffix_falls_back_to_octet_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.bin"
            path.write_bytes(b"\x00\x01")
            result = await read_local_file(path)
        self.assertEqual(result.content_type, "application/octet-stream")

    async def test_accepts_string_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.txt"
            path.write_bytes(b"hi")
            result = await read_local_file(str(path))
        self.assertEqual(result.bytes, b"hi")

    async def test_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            await read_local_file("/nonexistent/path/to/nothing.pdf")

    async def test_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(IsADirectoryError):
                await read_local_file(tmp)
