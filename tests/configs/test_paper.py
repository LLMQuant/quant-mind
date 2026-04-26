"""Tests for configs.paper."""

import unittest
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from quantmind.configs.paper import (
    ArxivIdentifier,
    DoiIdentifier,
    HttpUrl,
    LocalFilePath,
    PaperFlowCfg,
    PaperInput,
    RawText,
)


class PaperFlowCfgTests(unittest.TestCase):
    def test_defaults(self):
        cfg = PaperFlowCfg()
        self.assertEqual(cfg.model, "gpt-4o")
        self.assertTrue(cfg.extract_methodology)
        self.assertTrue(cfg.extract_limitations)
        self.assertIsNone(cfg.asset_class_hint)


class PaperInputDiscriminatedTests(unittest.TestCase):
    def setUp(self):
        self.adapter = TypeAdapter(PaperInput)

    def test_arxiv_round_trip(self):
        v = self.adapter.validate_python({"type": "arxiv", "id": "2604.12345"})
        self.assertIsInstance(v, ArxivIdentifier)
        self.assertEqual(v.id, "2604.12345")

    def test_http(self):
        v = self.adapter.validate_python(
            {"type": "http", "url": "https://example.com/p.pdf"}
        )
        self.assertIsInstance(v, HttpUrl)

    def test_local(self):
        v = self.adapter.validate_python(
            {"type": "local", "path": "/tmp/p.pdf"}
        )
        self.assertIsInstance(v, LocalFilePath)
        self.assertEqual(v.path, Path("/tmp/p.pdf"))

    def test_text(self):
        v = self.adapter.validate_python({"type": "text", "text": "hello"})
        self.assertIsInstance(v, RawText)

    def test_doi(self):
        v = self.adapter.validate_python({"type": "doi", "doi": "10.1000/xyz"})
        self.assertIsInstance(v, DoiIdentifier)

    def test_unknown_type_rejected(self):
        with self.assertRaises(ValidationError):
            self.adapter.validate_python({"type": "ftp", "url": "x"})


if __name__ == "__main__":
    unittest.main()
