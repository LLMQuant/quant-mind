"""Vectorless persistence tests for paper structure trees."""

import hashlib
import sqlite3
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from quantmind.knowledge import (
    ArtifactLocator,
    PaperArtifactKind,
    PaperStructureTree,
    TreeNode,
)
from quantmind.library import LocalKnowledgeLibrary, SemanticQuery
from tests.paper_helpers import build_paper_result, build_paper_structure_tree


class _FakeEmbeddingProvider:
    def __init__(self) -> None:
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
        return [[float(index + 1) for index in range(size)] for _ in texts]

    async def close(self) -> None:
        """Release no resources."""


class StructureTreeLibraryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temporary_directory.name) / "structure.db"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    async def _open(
        self,
        provider: _FakeEmbeddingProvider,
    ) -> LocalKnowledgeLibrary:
        return await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )

    async def test_put_reopen_get_and_idempotency_without_projections(
        self,
    ) -> None:
        result = build_paper_result()
        tree = build_paper_structure_tree()
        provider = _FakeEmbeddingProvider()
        library = await self._open(provider)
        await library.put_paper(result)
        await library.put_paper_structure_tree(tree)
        await library.put_paper_structure_tree(tree)
        await library.close()

        self.assertEqual(len(provider.calls), 1)
        with sqlite3.connect(self.db_path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 4)
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_artifacts").fetchone()[
                    0
                ],
                3,
            )
            self.assertEqual(
                db.execute(
                    "SELECT COUNT(*) FROM paper_artifact_members"
                ).fetchone()[0],
                len(result.chunk_set.chunks) + len(tree.nodes),
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_projections").fetchone()[
                    0
                ],
                len(result.chunk_set.chunks) + 1,
            )
            target_count = db.execute(
                "SELECT target_count FROM paper_artifacts WHERE artifact_id = ?",
                (str(tree.id),),
            ).fetchone()[0]
            self.assertEqual(target_count, 0)

        reopened = await self._open(_FakeEmbeddingProvider())
        try:
            restored = await reopened.get_artifact(tree.id)
            self.assertIsInstance(restored, PaperStructureTree)
            self.assertEqual(restored, tree)
            hits = await reopened.search(SemanticQuery(text="attention"))
            self.assertTrue(hits)
            self.assertNotIn(
                PaperArtifactKind.STRUCTURE_TREE,
                {hit.item_type for hit in hits},
            )
        finally:
            await reopened.close()

    async def test_resolve_node_assembles_cited_chunk_content(self) -> None:
        result = build_paper_result()
        tree = build_paper_structure_tree()
        library = await self._open(_FakeEmbeddingProvider())
        try:
            await library.put_paper(result)
            await library.put_paper_structure_tree(tree)
            node = next(
                value
                for value in tree.nodes.values()
                if value.title == "Attention and results"
            )
            resolved = await library.resolve(
                ArtifactLocator(
                    source_revision_id=tree.source_revision_id,
                    artifact_id=tree.id,
                    artifact_kind=PaperArtifactKind.STRUCTURE_TREE,
                    member_id=node.node_id,
                )
            )

            self.assertIsInstance(resolved, TreeNode)
            assert isinstance(resolved, TreeNode)
            self.assertEqual(
                resolved.content,
                "\n\n".join(
                    chunk.text for chunk in result.chunk_set.chunks[1:]
                ),
            )
            self.assertEqual(
                {citation.page for citation in resolved.citations}, {2}
            )
        finally:
            await library.close()

    async def test_put_requires_stored_source_and_chunk_set(self) -> None:
        tree = build_paper_structure_tree()
        library = await self._open(_FakeEmbeddingProvider())
        try:
            with self.assertRaises(KeyError):
                await library.put_paper_structure_tree(tree)
        finally:
            await library.close()

        with sqlite3.connect(self.db_path) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_artifacts").fetchone()[
                    0
                ],
                0,
            )

    async def test_rehydrate_fails_closed_on_member_metadata_drift(
        self,
    ) -> None:
        result = build_paper_result()
        tree = build_paper_structure_tree()
        library = await self._open(_FakeEmbeddingProvider())
        await library.put_paper(result)
        await library.put_paper_structure_tree(tree)
        await library.close()

        node = next(value for value in tree.nodes.values() if value.parent_id)
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                UPDATE paper_artifact_members SET position = 99
                WHERE artifact_id = ? AND member_id = ?
                """,
                (str(tree.id), str(node.node_id)),
            )

        reopened = await self._open(_FakeEmbeddingProvider())
        try:
            with self.assertRaisesRegex(
                RuntimeError, "member metadata mismatch"
            ):
                await reopened.get_artifact(tree.id)
        finally:
            await reopened.close()


class SchemaMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_version_three_adds_vectorless_and_parent_member_support(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3.db"
            with sqlite3.connect(path) as db:
                db.executescript(
                    """
                    CREATE TABLE paper_sources (
                        source_revision_id TEXT PRIMARY KEY
                    );
                    CREATE TABLE paper_artifacts (
                        artifact_id TEXT PRIMARY KEY,
                        source_revision_id TEXT NOT NULL,
                        artifact_kind TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        producer_config_hash TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        canonical_hash TEXT NOT NULL,
                        member_count INTEGER NOT NULL CHECK (member_count >= 0),
                        target_count INTEGER NOT NULL CHECK (target_count > 0),
                        UNIQUE (
                            source_revision_id,
                            artifact_kind,
                            producer_config_hash
                        )
                    );
                    CREATE TABLE paper_artifact_members (
                        artifact_id TEXT NOT NULL,
                        member_id TEXT NOT NULL,
                        position INTEGER NOT NULL CHECK (position >= 0),
                        payload_json TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        PRIMARY KEY (artifact_id, member_id),
                        UNIQUE (artifact_id, position)
                    );
                    CREATE INDEX paper_artifacts_source_kind
                        ON paper_artifacts(source_revision_id, artifact_kind);
                    PRAGMA user_version = 3;
                    """
                )

            library = await LocalKnowledgeLibrary.open(
                path,
                embedding_model="fake-2d",
                _embedding_provider=_FakeEmbeddingProvider(),
            )
            await library.close()

            with sqlite3.connect(path) as db:
                self.assertEqual(
                    db.execute("PRAGMA user_version").fetchone()[0],
                    4,
                )
                columns = {
                    row[1]
                    for row in db.execute(
                        "PRAGMA table_info(paper_artifact_members)"
                    ).fetchall()
                }
                self.assertIn("parent_id", columns)
                source_id = "source"
                db.execute(
                    "INSERT INTO paper_sources(source_revision_id) VALUES (?)",
                    (source_id,),
                )
                db.execute(
                    """
                    INSERT INTO paper_artifacts (
                        artifact_id, source_revision_id, artifact_kind,
                        schema_version, producer_config_hash, payload_json,
                        canonical_hash, member_count, target_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)
                    """,
                    (
                        "tree",
                        source_id,
                        PaperArtifactKind.STRUCTURE_TREE,
                        "1.0",
                        "a" * 64,
                        "{}",
                        hashlib.sha256(b"{}").hexdigest(),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
