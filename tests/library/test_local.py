import json
import sqlite3
import tempfile
import unittest
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from quantmind.knowledge import Citation, News, Paper, SourceRef, TreeNode
from quantmind.library import LocalKnowledgeLibrary, SemanticQuery


class _FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        dimension: int = 2,
        vectors: dict[str, list[float]] | None = None,
    ) -> None:
        self.dimension = dimension
        self.vectors = vectors or {}
        self.calls: list[tuple[str, int | None, tuple[str, ...]]] = []
        self.closed = False

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        self.calls.append((model, dimensions, tuple(texts)))
        output: list[list[float]] = []
        for text in texts:
            configured = self.vectors.get(text)
            if configured is not None:
                output.append(configured)
                continue
            size = dimensions or self.dimension
            seed = sum(
                (index + 1) * ord(char) for index, char in enumerate(text)
            )
            output.append(
                [
                    float(1 + ((seed >> (offset * 3)) % 17))
                    for offset in range(size)
                ]
            )
        return output

    async def close(self) -> None:
        self.closed = True


def _time(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 7, day, hour, tzinfo=timezone.utc)


def _news(
    headline: str,
    *,
    item_id: UUID | None = None,
    as_of: datetime | None = None,
    available_at: datetime | None = None,
    source_kind: str = "rss",
    source_hash: str | None = None,
    confidence: str = "medium",
    tags: list[str] | None = None,
    citations: list[Citation] | None = None,
) -> News:
    return News(
        id=item_id or uuid4(),
        as_of=as_of or _time(1),
        available_at=available_at,
        source=SourceRef(
            kind=source_kind,  # type: ignore[arg-type]
            uri=f"https://example.com/{headline}",
            content_hash=source_hash,
        ),
        confidence=confidence,  # type: ignore[arg-type]
        citations=citations or [],
        tags=tags or [],
        headline=headline,
        event_type="capital_expenditure",
        entities=["Alpha Corp"],
        timestamp=as_of or _time(1),
    )


def _paper() -> tuple[Paper, UUID, UUID]:
    root_id = uuid4()
    methods_id = uuid4()
    results_id = uuid4()
    root = TreeNode(
        node_id=root_id,
        title="Capital Spending Outlook",
        summary="Company-wide investment outlook",
        citations=[Citation(source_id="paper", page=1, node_id=root_id)],
        children_ids=[methods_id, results_id],
    )
    methods = TreeNode(
        node_id=methods_id,
        parent_id=root_id,
        title="Methodology",
        summary="Survey of management guidance",
        citations=[Citation(source_id="paper", page=2, node_id=methods_id)],
    )
    results = TreeNode(
        node_id=results_id,
        parent_id=root_id,
        position=1,
        title="Results",
        summary="Capital expenditure is expected to rise",
    )
    paper = Paper(
        as_of=_time(1),
        available_at=_time(2),
        source=SourceRef(kind="arxiv", content_hash="paper-v1"),
        confidence="high",
        tags=["macro", "rates", "capex"],
        citations=[Citation(source_id="paper", page=1)],
        root_node_id=root_id,
        nodes={root_id: root, methods_id: methods, results_id: results},
        arxiv_id="2607.00111",
    )
    return paper, methods_id, results_id


class LocalKnowledgeLibraryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temporary_directory.name) / "library.db"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    async def test_flat_put_get_and_deterministic_best_first_search(self):
        alpha = _news(
            "Alpha expands capacity",
            available_at=_time(2),
            tags=["capex"],
            citations=[Citation(source_id="rss:alpha", quote="capacity rises")],
        )
        beta = _news("Beta pauses investment", available_at=_time(2))
        query_text = "management increases capital expenditure"
        provider = _FakeEmbeddingProvider(
            vectors={
                alpha.embedding_text(): [1.0, 0.0],
                beta.embedding_text(): [0.0, 1.0],
                query_text: [1.0, 0.0],
            }
        )
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        try:
            await library.put(beta)
            await library.put(alpha)
            hits = await library.search(SemanticQuery(text=query_text, top_k=2))
            self.assertEqual([hit.item_id for hit in hits], [alpha.id, beta.id])
            self.assertAlmostEqual(hits[0].score, 1.0)
            self.assertIsNone(hits[0].node_id)
            self.assertEqual(hits[0].matched_text, alpha.embedding_text())
            self.assertEqual(hits[0].source, alpha.source)
            self.assertEqual(hits[0].citations, alpha.citations)
            self.assertEqual(hits[0].available_at, alpha.available_at)
            self.assertEqual(await library.get(alpha.id), alpha)
        finally:
            await library.close()

    async def test_tree_root_and_non_root_nodes_use_exact_grain_and_filters(
        self,
    ):
        paper, methods_id, results_id = _paper()
        unrelated = _news("Unrelated event", tags=["macro", "rates"])
        query_text = "capital expenditure outlook"
        provider = _FakeEmbeddingProvider(
            vectors={
                paper.embedding_text(): [1.0, 0.0],
                paper.nodes[methods_id].embedding_text(): [0.8, 0.2],
                paper.nodes[results_id].embedding_text(): [0.9, 0.1],
                unrelated.embedding_text(): [0.0, 1.0],
                query_text: [1.0, 0.0],
            }
        )
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        try:
            await library.put(unrelated)
            await library.put(paper)
            with sqlite3.connect(self.db_path) as db:
                root_row = db.execute(
                    """
                    SELECT item_shape, payload_json, node_count
                    FROM knowledge_items WHERE item_id = ?
                    """,
                    (str(paper.id),),
                ).fetchone()
                assert root_row is not None
                self.assertEqual(root_row[0], "tree")
                self.assertNotIn("nodes", json.loads(root_row[1]))
                self.assertEqual(root_row[2], 3)
                self.assertEqual(
                    db.execute(
                        "SELECT COUNT(*) FROM knowledge_nodes WHERE item_id = ?",
                        (str(paper.id),),
                    ).fetchone()[0],
                    3,
                )
            hits = await library.search(
                SemanticQuery(
                    text=query_text,
                    item_types=["paper"],
                    source_kinds=["arxiv"],
                    confidence="high",
                    tags=["macro", "rates"],
                    tree_id=paper.id,
                    as_of_before=_time(1),
                    available_at_before=_time(2),
                    top_k=10,
                )
            )
            self.assertEqual(len(hits), 3)
            self.assertEqual(sum(hit.node_id is None for hit in hits), 1)
            self.assertEqual(
                {hit.node_id for hit in hits if hit.node_id is not None},
                {methods_id, results_id},
            )
            root_hit = next(hit for hit in hits if hit.node_id is None)
            self.assertEqual(root_hit.matched_text, paper.embedding_text())
            self.assertEqual(root_hit.citations, paper.root().citations)
            methods_hit = next(hit for hit in hits if hit.node_id == methods_id)
            self.assertEqual(
                methods_hit.matched_text,
                paper.nodes[methods_id].embedding_text(),
            )
            self.assertEqual(
                methods_hit.citations,
                paper.nodes[methods_id].citations,
            )
            stored = await library.get(methods_hit.item_id)
            self.assertIsInstance(stored, Paper)
            assert isinstance(stored, Paper)
            self.assertEqual(
                stored.find_path(methods_id)[-1].node_id, methods_id
            )
        finally:
            await library.close()

    async def test_as_of_and_availability_cutoffs_are_distinct(self):
        safe = _news(
            "Known after cutoff",
            as_of=_time(1),
            available_at=_time(3),
        )
        unknown = _news("Unknown availability", as_of=_time(1))
        later_information = _news(
            "Later information available early",
            as_of=_time(5),
            available_at=_time(2),
        )
        provider = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        try:
            for item in (safe, unknown, later_information):
                await library.put(item)
            as_of_hits = await library.search(
                SemanticQuery(text="cutoff", as_of_before=_time(2), top_k=10)
            )
            self.assertEqual(
                {hit.item_id for hit in as_of_hits},
                {safe.id, unknown.id},
            )
            availability_hits = await library.search(
                SemanticQuery(
                    text="cutoff",
                    available_at_before=_time(2, 12),
                    top_k=10,
                )
            )
            self.assertEqual(
                {hit.item_id for hit in availability_hits},
                {later_information.id},
            )
            combined = await library.search(
                SemanticQuery(
                    text="cutoff",
                    as_of_before=_time(2),
                    available_at_before=_time(4),
                    top_k=10,
                )
            )
            self.assertEqual([hit.item_id for hit in combined], [safe.id])
        finally:
            await library.close()

    async def test_reput_is_idempotent_and_invalidates_only_changed_metadata(
        self,
    ):
        item = _news(
            "Original headline",
            source_hash="source-v1",
            tags=["old"],
        )
        provider = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        try:
            await library.put(item)
            self.assertEqual(len(provider.calls), 1)
            await library.put(item)
            self.assertEqual(len(provider.calls), 1)

            metadata_only = item.model_copy(update={"tags": ["new"]})
            await library.put(metadata_only)
            self.assertEqual(len(provider.calls), 1)

            changed_text = metadata_only.model_copy(
                update={"headline": "Changed headline"}
            )
            await library.put(changed_text)
            self.assertEqual(
                provider.calls[-1][2], (changed_text.embedding_text(),)
            )

            changed_source = changed_text.model_copy(
                update={
                    "source": changed_text.source.model_copy(
                        update={"content_hash": "source-v2"}
                    )
                }
            )
            await library.put(changed_source)
            self.assertEqual(len(provider.calls), 3)

            changed_schema = changed_source.model_copy(
                update={"schema_version": "1.1"}
            )
            await library.put(changed_schema)
            self.assertEqual(len(provider.calls), 4)
        finally:
            await library.close()

    async def test_tree_projection_change_invalidates_only_one_node(self):
        paper, methods_id, _ = _paper()
        provider = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        try:
            await library.put(paper)
            self.assertEqual(len(provider.calls[0][2]), 3)
            changed_node = paper.nodes[methods_id].model_copy(
                update={"summary": "A newly described survey method"}
            )
            changed_nodes = dict(paper.nodes)
            changed_nodes[methods_id] = changed_node
            changed_paper = paper.model_copy(update={"nodes": changed_nodes})
            await library.put(changed_paper)
            self.assertEqual(
                provider.calls[-1][2], (changed_node.embedding_text(),)
            )

            schema_change = changed_paper.model_copy(
                update={"schema_version": "1.1"}
            )
            await library.put(schema_change)
            self.assertEqual(len(provider.calls[-1][2]), 3)
        finally:
            await library.close()

    async def test_model_and_dimension_changes_have_explicit_stale_rebuild_path(
        self,
    ):
        item = _news("Model metadata")
        first = _FakeEmbeddingProvider(dimension=2)
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="model-a",
            embedding_dimensions=2,
            _embedding_provider=first,
        )
        await library.put(item)
        await library.close()

        second = _FakeEmbeddingProvider(dimension=2)
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="model-b",
            embedding_dimensions=2,
            _embedding_provider=second,
        )
        with self.assertRaisesRegex(RuntimeError, "Stale index data.*model"):
            await library.search(SemanticQuery(text="query"))
        self.assertEqual(second.calls, [])
        await library.put(item)
        self.assertEqual(len(second.calls), 1)
        await library.close()

        third = _FakeEmbeddingProvider(dimension=3)
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="model-b",
            embedding_dimensions=3,
            _embedding_provider=third,
        )
        try:
            await library.put(item)
            self.assertEqual(third.calls[0][1], 3)
            hits = await library.search(SemanticQuery(text="query"))
            self.assertEqual([hit.item_id for hit in hits], [item.id])
        finally:
            await library.close()

    async def test_index_rebuilds_from_sqlite_without_reembedding_items(self):
        item = _news("Persistent item", available_at=_time(2))
        first = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=first,
        )
        await library.put(item)
        await library.close()

        second = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=second,
        )
        try:
            self.assertEqual(await library.get(item.id), item)
            self.assertEqual(second.calls, [])
            hits = await library.search(SemanticQuery(text="persistent"))
            self.assertEqual([hit.item_id for hit in hits], [item.id])
            self.assertEqual(second.calls[0][2], ("persistent",))
        finally:
            await library.close()

    async def test_delete_removes_canonical_root_and_nodes_transactionally(
        self,
    ):
        paper, _, _ = _paper()
        provider = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        try:
            await library.put(paper)
            await library.delete(paper.id)
            with self.assertRaises(KeyError):
                await library.get(paper.id)
            with self.assertRaises(KeyError):
                await library.delete(paper.id)
            calls_before_search = len(provider.calls)
            self.assertEqual(
                await library.search(SemanticQuery(text="query")), []
            )
            self.assertEqual(len(provider.calls), calls_before_search)
        finally:
            await library.close()
        with sqlite3.connect(self.db_path) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[
                    0
                ],
                0,
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM semantic_records").fetchone()[
                    0
                ],
                0,
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM knowledge_nodes").fetchone()[
                    0
                ],
                0,
            )

    async def test_stale_canonical_get_fails_but_delete_can_recover(self):
        item = _news("Stale canonical")
        provider = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        try:
            await library.put(item)
            with sqlite3.connect(self.db_path) as db:
                db.execute(
                    "UPDATE knowledge_items SET payload_json = '{}' "
                    "WHERE item_id = ?",
                    (str(item.id),),
                )
            with self.assertRaisesRegex(
                RuntimeError, "Stale canonical knowledge"
            ):
                await library.get(item.id)
            await library.delete(item.id)
            with self.assertRaises(KeyError):
                await library.get(item.id)
        finally:
            await library.close()

    async def test_orphan_and_missing_derived_data_are_reported_as_stale(self):
        item = _news("Orphan target")
        provider = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        await library.put(item)
        await library.close()
        with sqlite3.connect(self.db_path) as db:
            db.execute("PRAGMA foreign_keys = OFF")
            db.execute(
                "DELETE FROM knowledge_items WHERE item_id = ?", (str(item.id),)
            )

        reopened_provider = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=reopened_provider,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "Stale data"):
                await library.get(item.id)
            with self.assertRaisesRegex(RuntimeError, "Stale data"):
                await library.delete(item.id)
            with self.assertRaisesRegex(RuntimeError, "Stale index data"):
                await library.search(SemanticQuery(text="query"))
        finally:
            await library.close()

    async def test_corrupt_canonical_tree_node_fails_rehydration(self):
        paper, methods_id, _ = _paper()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=_FakeEmbeddingProvider(),
        )
        try:
            await library.put(paper)
            with sqlite3.connect(self.db_path) as db:
                db.execute(
                    """
                    UPDATE knowledge_nodes SET payload_json = '{}'
                    WHERE item_id = ? AND node_id = ?
                    """,
                    (str(paper.id), str(methods_id)),
                )
            with self.assertRaisesRegex(
                RuntimeError, "node.*content hash mismatch"
            ):
                await library.get(paper.id)
        finally:
            await library.close()

    async def test_corrupt_vector_and_query_dimension_mismatch_fail_clearly(
        self,
    ):
        item = _news("Corrupt vector")
        provider = _FakeEmbeddingProvider()
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=provider,
        )
        await library.put(item)
        await library.close()
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "UPDATE semantic_records SET embedding = ? WHERE item_id = ?",
                (b"bad", str(item.id)),
            )

        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=_FakeEmbeddingProvider(),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "Corrupt index data"):
                await library.search(SemanticQuery(text="query"))
        finally:
            await library.close()

        mismatch_path = Path(self._temporary_directory.name) / "mismatch.db"
        library = await LocalKnowledgeLibrary.open(
            mismatch_path,
            embedding_model="fake",
            embedding_dimensions=2,
            _embedding_provider=_FakeEmbeddingProvider(dimension=2),
        )
        await library.put(item)
        await library.close()
        library = await LocalKnowledgeLibrary.open(
            mismatch_path,
            embedding_model="fake",
            _embedding_provider=_FakeEmbeddingProvider(dimension=3),
        )
        try:
            with self.assertRaisesRegex(ValueError, "dimension mismatch"):
                await library.search(SemanticQuery(text="query"))
        finally:
            await library.close()

    async def test_closed_library_rejects_operations(self):
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="fake-2d",
            _embedding_provider=_FakeEmbeddingProvider(),
        )
        await library.close()
        await library.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await library.get(uuid4())


if __name__ == "__main__":
    unittest.main()
