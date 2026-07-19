"""Offline source/artifact/projection tests for paper library persistence."""

import hashlib
import sqlite3
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from quantmind.knowledge import (
    PaperArtifactKind,
    PaperChunk,
    PaperGlobalSummary,
)
from quantmind.library import LocalKnowledgeLibrary, SemanticQuery
from tests.paper_helpers import build_paper_result


class _FakeEmbeddingProvider:
    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[tuple[str, ...]] = []

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        del model
        self.calls.append(tuple(texts))
        size = dimensions or 2
        return [
            self.vectors.get(
                text,
                [
                    float((sum(map(ord, text)) + offset) % 17 + 1)
                    for offset in range(size)
                ],
            )
            for text in texts
        ]

    async def close(self) -> None:
        """Release no resources."""


class _FailingEmbeddingProvider(_FakeEmbeddingProvider):
    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        del texts, model, dimensions
        raise RuntimeError("embedding unavailable")


class PaperLibraryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temporary_directory.name) / "paper.db"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    async def test_put_persists_explicit_source_artifact_and_projection_layers(
        self,
    ) -> None:
        result = build_paper_result()
        provider = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        try:
            await library.put_paper(result)
        finally:
            await library.close()

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(provider.calls[0]), 4)
        with sqlite3.connect(self.db_path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 4)
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_sources").fetchone()[0],
                1,
            )
            self.assertEqual(
                db.execute(
                    "SELECT COUNT(*) FROM paper_source_assets"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_artifacts").fetchone()[
                    0
                ],
                2,
            )
            self.assertEqual(
                db.execute(
                    "SELECT COUNT(*) FROM paper_artifact_members"
                ).fetchone()[0],
                3,
            )
            self.assertEqual(
                db.execute(
                    "SELECT COUNT(*) FROM paper_artifact_lineage"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_projections").fetchone()[
                    0
                ],
                4,
            )
            payloads = [
                row[0]
                for row in db.execute(
                    "SELECT payload_json FROM paper_artifacts"
                ).fetchall()
            ]
            self.assertTrue(
                all("embedding" not in payload for payload in payloads)
            )

    async def test_reopen_round_trip_reuses_vectors_and_resolves_hits(
        self,
    ) -> None:
        result = build_paper_result()
        multi_head = result.chunk_set.chunks[1]
        summary_query = "What is the paper's central contribution?"
        chunk_query = "How does multi-head attention work?"
        first = _FakeEmbeddingProvider(
            vectors={
                result.global_summary.summary: [1.0, 0.0],
                multi_head.text: [1.0, 0.0],
                result.chunk_set.chunks[0].text: [0.0, 1.0],
                result.chunk_set.chunks[2].text: [0.0, 1.0],
            }
        )
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=first,
        )
        await library.put_paper(result)
        await library.close()

        second = _FakeEmbeddingProvider(
            vectors={
                result.global_summary.summary: [1.0, 0.0],
                multi_head.text: [1.0, 0.0],
                summary_query: [1.0, 0.0],
                chunk_query: [1.0, 0.0],
            }
        )
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=second,
        )
        try:
            restored = await library.get_paper(result.source_revision.id)
            self.assertEqual(restored.chunk_set, result.chunk_set)
            self.assertEqual(restored.global_summary, result.global_summary)
            self.assertEqual(
                restored.source_revision.blob_for(
                    restored.source_revision.raw_asset_id
                ),
                result.source_revision.blob_for(
                    result.source_revision.raw_asset_id
                ),
            )
            self.assertEqual(second.calls, [])

            summary_hits = await library.search(
                SemanticQuery(
                    text=summary_query,
                    artifact_kinds=[PaperArtifactKind.GLOBAL_SUMMARY],
                    top_k=3,
                )
            )
            self.assertEqual(len(summary_hits), 1)
            self.assertEqual(
                summary_hits[0].locator.artifact_id,
                result.global_summary.id,
            )
            self.assertEqual(summary_hits[0].projection.model, "fake-2d")
            summary = await library.resolve(summary_hits[0].locator)
            self.assertIsInstance(summary, PaperGlobalSummary)

            chunk_hits = await library.search(
                SemanticQuery(
                    text=chunk_query,
                    artifact_kinds=[PaperArtifactKind.CHUNK_SET],
                    top_k=5,
                )
            )
            self.assertIn(
                multi_head.chunk_id, [hit.node_id for hit in chunk_hits]
            )
            matching_hit = next(
                hit for hit in chunk_hits if hit.node_id == multi_head.chunk_id
            )
            chunk = await library.resolve(matching_hit.locator)
            self.assertIsInstance(chunk, PaperChunk)
            assert isinstance(chunk, PaperChunk)
            self.assertEqual(chunk.source_spans[0].page_number, 2)
            self.assertEqual(matching_hit.citations[0].page, 2)
        finally:
            await library.close()

    async def test_reput_and_changed_summary_selectively_rebuild_projections(
        self,
    ) -> None:
        original = build_paper_result()
        changed = build_paper_result(summary_text="A refreshed cited summary.")
        provider = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        try:
            await library.put_paper(original)
            self.assertEqual(len(provider.calls), 1)
            await library.put_paper(original)
            self.assertEqual(len(provider.calls), 1)

            await library.put_paper(changed)
            self.assertEqual(
                provider.calls[-1], (changed.global_summary.summary,)
            )
            self.assertEqual(len(provider.calls), 2)
        finally:
            await library.close()

    async def test_multiple_chunk_and_summary_versions_coexist(self) -> None:
        first = build_paper_result(chunk_size=128)
        second = build_paper_result(chunk_size=256)
        provider = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        try:
            await library.put_paper(first)
            await library.put_paper(second)
            with self.assertRaisesRegex(ValueError, "specify an artifact ID"):
                await library.get_paper(first.source_revision.id)
            restored = await library.get_paper(
                first.source_revision.id,
                chunk_set_id=second.chunk_set.id,
                summary_id=second.global_summary.id,
            )
            self.assertEqual(restored.chunk_set.id, second.chunk_set.id)
            self.assertEqual(
                restored.global_summary.id, second.global_summary.id
            )
        finally:
            await library.close()
        with sqlite3.connect(self.db_path) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_artifacts").fetchone()[
                    0
                ],
                4,
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_projections").fetchone()[
                    0
                ],
                8,
            )

    async def test_required_projection_failure_is_atomic(self) -> None:
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="failed",
            embedding_dimensions=2,
            _embedding_provider=_FailingEmbeddingProvider(),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
                await library.put_paper(build_paper_result())
        finally:
            await library.close()
        with sqlite3.connect(self.db_path) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_sources").fetchone()[0],
                0,
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_artifacts").fetchone()[
                    0
                ],
                0,
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_projections").fetchone()[
                    0
                ],
                0,
            )

    async def test_rehydrate_rejects_asset_metadata_drift(self) -> None:
        result = build_paper_result()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=_FakeEmbeddingProvider(),
        )
        await library.put_paper(result)
        await library.close()

        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "UPDATE paper_source_assets SET media_type = ?",
                ("application/tampered",),
            )

        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=_FakeEmbeddingProvider(),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                await library.get_paper(result.source_revision.id)
        finally:
            await library.close()

    async def test_rehydrate_rejects_missing_summary_lineage(self) -> None:
        result = build_paper_result()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=_FakeEmbeddingProvider(),
        )
        await library.put_paper(result)
        await library.close()

        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "DELETE FROM paper_artifact_lineage WHERE artifact_id = ?",
                (str(result.global_summary.id),),
            )

        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=_FakeEmbeddingProvider(),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "lineage mismatch"):
                await library.get_artifact(result.global_summary.id)
        finally:
            await library.close()

    async def test_search_rejects_projection_text_drift(self) -> None:
        result = build_paper_result()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=_FakeEmbeddingProvider(),
        )
        await library.put_paper(result)
        await library.close()

        tampered = "tampered summary projection"
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                UPDATE paper_projections
                SET matched_text = ?, projection_hash = ?
                WHERE artifact_kind = 'paper_summary'
                """,
                (tampered, hashlib.sha256(tampered.encode()).hexdigest()),
            )

        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=_FakeEmbeddingProvider(),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "canonical text"):
                await library.search(SemanticQuery(text="summary"))
        finally:
            await library.close()


if __name__ == "__main__":
    unittest.main()
