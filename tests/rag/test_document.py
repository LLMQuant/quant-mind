"""Tests for page-aware LlamaIndex document RAG."""

import unittest
from pathlib import Path

from quantmind.preprocess.format import parse_pdf
from quantmind.rag import (
    SentenceSplitterConfig,
    chunk_parsed_document,
    retrieve_parsed_document,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_TINY = (
    Path(__file__).resolve().parents[1] / "preprocess" / "fixtures" / "tiny.pdf"
)
_GOLDEN = _FIXTURES / "paper" / "golden" / "paper.pdf"


class DocumentRagTests(unittest.IsolatedAsyncioTestCase):
    async def test_chunks_and_bm25_hits_keep_page_evidence(self):
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

    async def test_retrieval_rejects_invalid_query_arguments(self):
        document = await parse_pdf(_TINY.read_bytes())
        chunks = chunk_parsed_document(document)
        with self.assertRaisesRegex(ValueError, "query"):
            retrieve_parsed_document(chunks, "   ")
        with self.assertRaisesRegex(ValueError, "top_k"):
            retrieve_parsed_document(chunks, "fixture", top_k=0)
