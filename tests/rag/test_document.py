"""Tests for page-aware LlamaIndex document RAG."""

import unittest
from pathlib import Path

from quantmind.preprocess.format import parse_pdf
from quantmind.rag import (
    ParsedChunk,
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
        self.assertTrue(
            all(0 <= chunk.start_char < chunk.end_char for chunk in chunks)
        )
        repeated = chunk_parsed_document(
            document,
            config=SentenceSplitterConfig(chunk_size=256, chunk_overlap=32),
        )
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [chunk.chunk_id for chunk in repeated],
        )

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


class ScriptAwareRetrievalTests(unittest.TestCase):
    """Retrieval over scripts that are not whitespace-delimited."""

    @staticmethod
    def _chunks(texts: tuple[str, ...]) -> tuple[ParsedChunk, ...]:
        return tuple(
            ParsedChunk(
                chunk_id=f"chunk-{index}",
                text=text,
                source_hash="0" * 64,
                page_number=index + 1,
                start_char=0,
                end_char=len(text),
                block_boxes=(),
                screenshot_path=None,
                image_paths=(),
            )
            for index, text in enumerate(texts)
        )

    def test_japanese_query_ranks_the_matching_chunk_first(self):
        chunks = self._chunks(
            (
                "国際標準化に関する動向と規格策定プロセスの概観",
                "人材育成の方針と研修体系の整備について",
            )
        )

        hits = retrieve_parsed_document(chunks, "国際標準化", top_k=2)

        self.assertEqual(hits[0].chunk.chunk_id, "chunk-0")
        self.assertGreater(hits[0].score, 0.0)

    def test_chinese_query_ranks_the_matching_chunk_first(self):
        chunks = self._chunks(
            (
                "货币政策与利率变动对债券市场的影响",
                "上市公司季度财报披露时间表",
            )
        )

        hits = retrieve_parsed_document(chunks, "利率变动", top_k=2)

        self.assertEqual(hits[0].chunk.chunk_id, "chunk-0")
        self.assertGreater(hits[0].score, 0.0)

    def test_latin_query_ranks_the_matching_chunk_first(self):
        chunks = self._chunks(
            (
                "knowledge extraction architecture and retrieval evaluation",
                "deterministic preprocessing keeps provenance replayable",
            )
        )

        hits = retrieve_parsed_document(chunks, "retrieval evaluation", top_k=2)

        self.assertEqual(hits[0].chunk.chunk_id, "chunk-0")
        self.assertGreater(hits[0].score, 0.0)
