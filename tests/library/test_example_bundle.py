import json
import tempfile
import unittest
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from quantmind.knowledge import (
    BaseKnowledge,
    Earnings,
    News,
    Paper,
    TreeKnowledge,
)
from quantmind.library import LocalKnowledgeLibrary, SemanticQuery

_BUNDLE_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "library"
    / "data"
    / "ai_infrastructure.json"
)
_KNOWLEDGE_TYPES: dict[str, type[BaseKnowledge]] = {
    "earnings": Earnings,
    "news": News,
    "paper": Paper,
}


class _DeterministicEmbeddingProvider:
    """Provide stable local vectors so the example scenario stays offline."""

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        del model
        if dimensions != 2:
            raise ValueError("This test provider requires two dimensions")
        return [[1.0, float(1 + sum(map(ord, text)) % 97)] for text in texts]

    async def close(self) -> None:
        """Release no resources."""


class ExampleBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        bundle = json.loads(_BUNDLE_PATH.read_text(encoding="utf-8"))
        self.scenario = str(bundle["scenario"])
        self.items = [
            _KNOWLEDGE_TYPES[str(payload["item_type"])].model_validate(payload)
            for payload in bundle["items"]
        ]

    def test_bundle_is_a_concrete_cross_shape_scenario(self):
        self.assertIn("AI", self.scenario)
        self.assertEqual(
            {type(item) for item in self.items},
            {News, Earnings, Paper},
        )
        self.assertTrue(
            all("ai-infrastructure" in item.tags for item in self.items)
        )

    def test_sources_and_financial_times_are_auditable(self):
        for item in self.items:
            self.assertIsNotNone(item.source.uri)
            assert item.source.uri is not None
            self.assertTrue(item.source.uri.startswith("https://"))
            self.assertIsNotNone(item.as_of.utcoffset())
            self.assertIsNotNone(item.available_at)
            assert item.available_at is not None
            self.assertIsNotNone(item.available_at.utcoffset())
            self.assertTrue(item.citations)

    def test_paper_tree_has_stable_valid_navigation(self):
        paper = next(item for item in self.items if isinstance(item, Paper))
        self.assertEqual(len(paper.nodes), 4)
        self.assertEqual(
            [node.position for node in paper.children_of(paper.root_node_id)],
            [0, 1, 2],
        )
        self.assertEqual(len(list(paper.walk_dfs())), 4)
        for node_id, node in paper.nodes.items():
            self.assertIsInstance(node_id, UUID)
            self.assertEqual(node_id, node.node_id)
            path = paper.find_path(node_id)
            self.assertEqual(path[0].node_id, paper.root_node_id)
            self.assertEqual(path[-1].node_id, node_id)

    def test_bundle_creates_six_documented_semantic_targets(self):
        target_count = sum(
            len(item.nodes) if isinstance(item, TreeKnowledge) else 1
            for item in self.items
        )
        self.assertEqual(target_count, 6)


class ExampleBundleSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        bundle = json.loads(_BUNDLE_PATH.read_text(encoding="utf-8"))
        self.items = [
            _KNOWLEDGE_TYPES[str(payload["item_type"])].model_validate(payload)
            for payload in bundle["items"]
        ]
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temporary_directory.name) / "library.db"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    async def test_bundle_put_reopen_search_and_get(self):
        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="deterministic-2d",
            embedding_dimensions=2,
            _embedding_provider=_DeterministicEmbeddingProvider(),
        )
        for item in self.items:
            await library.put(item)
        await library.close()

        library = await LocalKnowledgeLibrary.open(
            self.db_path,
            embedding_model="deterministic-2d",
            embedding_dimensions=2,
            _embedding_provider=_DeterministicEmbeddingProvider(),
        )
        try:
            hits = await library.search(
                SemanticQuery(
                    text=(
                        "What evidence shows demand for AI infrastructure "
                        "is expanding?"
                    ),
                    source_kinds=["http", "arxiv"],
                    tags=["ai-infrastructure"],
                    available_at_before=datetime(
                        2026, 1, 1, tzinfo=timezone.utc
                    ),
                    top_k=10,
                )
            )
            self.assertEqual(len(hits), 6)
            self.assertEqual(
                {hit.item_type for hit in hits},
                {"news", "earnings", "paper"},
            )
            self.assertEqual(sum(hit.node_id is not None for hit in hits), 3)
            for item in self.items:
                self.assertEqual(await library.get(item.id), item)
        finally:
            await library.close()


if __name__ == "__main__":
    unittest.main()
