"""Vectorless persistence tests for paper structure trees."""

import hashlib
import sqlite3
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

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
        await library.put_paper_structure_tree(result.source_revision, tree)
        await library.put_paper_structure_tree(result.source_revision, tree)
        await library.close()

        self.assertEqual(provider.calls, [])
        with sqlite3.connect(self.db_path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 4)
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_artifacts").fetchone()[
                    0
                ],
                1,
            )
            self.assertEqual(
                db.execute(
                    "SELECT COUNT(*) FROM paper_artifact_members"
                ).fetchone()[0],
                len(tree.nodes),
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM paper_projections").fetchone()[
                    0
                ],
                0,
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
            # The reopened tree is an identical self-contained value: every
            # leaf node keeps its own text through dump/load, with no chunk
            # set present.
            self.assertEqual(restored, tree)
            assert isinstance(restored, PaperStructureTree)
            for node in restored.nodes.values():
                if node.children_ids:
                    self.assertIsNone(node.content)
                else:
                    self.assertTrue(node.content)
                    self.assertEqual(
                        node.content, tree.nodes[node.node_id].content
                    )
            hits = await reopened.search(SemanticQuery(text="attention"))
            self.assertEqual(hits, [])
            self.assertNotIn(
                PaperArtifactKind.STRUCTURE_TREE,
                {hit.item_type for hit in hits},
            )
        finally:
            await reopened.close()

    async def test_open_structure_returns_self_contained_tree(self) -> None:
        result = build_paper_result()
        tree = build_paper_structure_tree()
        library = await self._open(_FakeEmbeddingProvider())
        try:
            await library.put_paper_structure_tree(result.source_revision, tree)
            loaded = await library.open_structure(tree.id)
            self.assertIsInstance(loaded, PaperStructureTree)
            self.assertEqual(loaded, tree)
            leaf = next(
                node for node in loaded.nodes.values() if not node.children_ids
            )
            self.assertTrue(leaf.content)
        finally:
            await library.close()

    async def test_resolve_node_returns_stored_content_without_refill(
        self,
    ) -> None:
        result = build_paper_result()
        tree = build_paper_structure_tree()
        library = await self._open(_FakeEmbeddingProvider())
        try:
            await library.put_paper_structure_tree(
                result.source_revision,
                tree,
            )
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
            # resolve returns the node's own stored content verbatim; it does
            # not reconstruct text from source pages.
            self.assertEqual(resolved, node)
            self.assertEqual(resolved.content, node.content)
            self.assertEqual(
                resolved.content,
                result.source_revision.parsed.pages[1].text,
            )
            self.assertEqual(
                {citation.page for citation in resolved.citations}, {2}
            )
        finally:
            await library.close()

    async def test_put_rejects_a_tree_for_another_source(self) -> None:
        result = build_paper_result()
        tree = build_paper_structure_tree()
        mismatched = tree.model_copy(update={"source_revision_id": uuid4()})
        library = await self._open(_FakeEmbeddingProvider())
        try:
            with self.assertRaisesRegex(ValueError, "another source"):
                await library.put_paper_structure_tree(
                    result.source_revision,
                    mismatched,
                )
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
        await library.put_paper_structure_tree(result.source_revision, tree)
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
