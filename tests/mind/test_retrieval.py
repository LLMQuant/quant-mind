"""Offline tests for vectorless reasoning-based retrieval."""

import asyncio
import json
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from quantmind.configs import RetrievalCfg
from quantmind.knowledge import (
    ArtifactLocator,
    PaperArtifactKind,
    StructureTree,
)
from quantmind.library import LocalKnowledgeLibrary
from quantmind.mind import RetrievalError, retrieve
from quantmind.mind.retrieval import _serialize_structure
from tests.paper_helpers import build_paper_result, build_paper_structure_tree


class _FakeEmbeddingProvider:
    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        del model
        size = dimensions or 2
        return [[float(index + 1) for index in range(size)] for _ in texts]

    async def close(self) -> None:
        """Release no resources."""


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        path = Path(self._temporary_directory.name) / "retrieval.db"
        self.result = build_paper_result()
        self.tree = build_paper_structure_tree()
        self.library = await LocalKnowledgeLibrary.open(
            path,
            embedding_model="fake-2d",
            embedding_dimensions=2,
            _embedding_provider=_FakeEmbeddingProvider(),
        )
        await self.library.put_paper(self.result)
        await self.library.put_paper_structure_tree(self.tree)
        self.node = next(
            value
            for value in self.tree.nodes.values()
            if value.title == "Attention and results"
        )

    async def asyncTearDown(self) -> None:
        await self.library.close()
        self._temporary_directory.cleanup()

    async def test_single_pass_makes_one_call_and_resolves_page_evidence(
        self,
    ) -> None:
        fake_result = SimpleNamespace(
            final_output={"node_ids": [str(self.node.node_id)]}
        )
        run_mock = AsyncMock(return_value=fake_result)
        cfg = RetrievalCfg(model="litellm/anthropic/test-model")

        with patch("quantmind.mind.retrieval.Runner.run", new=run_mock):
            evidence = await retrieve(
                self.tree,
                "How does multi-head attention work?",
                library=self.library,
                cfg=cfg,
            )

        run_mock.assert_awaited_once()
        agent = run_mock.await_args.args[0]
        self.assertEqual(agent.model, cfg.model)
        self.assertEqual(run_mock.await_args.kwargs["max_turns"], 1)
        model_input = run_mock.await_args.args[1]
        self.assertNotIn(self.result.chunk_set.chunks[1].text, model_input)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].locator.member_id, self.node.node_id)
        self.assertEqual(evidence[0].title, "Attention and results")
        self.assertIn("Multi-head attention", evidence[0].content)
        self.assertEqual(
            {citation.page for citation in evidence[0].citations},
            {2},
        )

    async def test_agentic_grain_exposes_exactly_two_tools(self) -> None:
        async def fake_run(agent, input_value, **kwargs):
            del input_value, kwargs
            content_tool = next(
                tool for tool in agent.tools if tool.name == "get_node_content"
            )
            content = await content_tool.on_invoke_tool(
                MagicMock(),
                json.dumps({"node_ids": [str(self.node.node_id)]}),
            )
            self.assertIn("Multi-head attention", content)
            return SimpleNamespace(
                final_output={"node_ids": [str(self.node.node_id)]}
            )

        run_mock = AsyncMock(side_effect=fake_run)
        cfg = RetrievalCfg(grain="agentic", max_turns=6)

        with patch("quantmind.mind.retrieval.Runner.run", new=run_mock):
            evidence = await retrieve(
                self.tree,
                "Find the attention evidence.",
                library=self.library,
                cfg=cfg,
            )

        agent = run_mock.await_args.args[0]
        self.assertEqual(
            {tool.name for tool in agent.tools},
            {"get_document_structure", "get_node_content"},
        )
        self.assertEqual(run_mock.await_args.kwargs["max_turns"], 6)
        self.assertEqual(evidence[0].locator.member_id, self.node.node_id)

    async def test_matching_seed_is_forwarded_but_mismatch_is_rejected(
        self,
    ) -> None:
        seed = ArtifactLocator(
            source_revision_id=self.tree.source_revision_id,
            artifact_id=self.tree.id,
            artifact_kind=PaperArtifactKind.STRUCTURE_TREE,
            member_id=self.node.node_id,
        )
        run_mock = AsyncMock(
            return_value=SimpleNamespace(
                final_output={"node_ids": [str(self.node.node_id)]}
            )
        )
        with patch("quantmind.mind.retrieval.Runner.run", new=run_mock):
            await retrieve(
                self.tree,
                "Find attention.",
                library=self.library,
                cfg=RetrievalCfg(),
                seed_locators=[seed],
            )
        payload = json.loads(run_mock.await_args.args[1])
        self.assertEqual(payload["seed_node_ids"], [str(self.node.node_id)])

        mismatched = seed.model_copy(update={"artifact_id": uuid4()})
        with patch(
            "quantmind.mind.retrieval.Runner.run",
            new=AsyncMock(),
        ) as rejected_run:
            with self.assertRaisesRegex(ValueError, "selected structure tree"):
                await retrieve(
                    self.tree,
                    "Find attention.",
                    library=self.library,
                    cfg=RetrievalCfg(),
                    seed_locators=[mismatched],
                )
        rejected_run.assert_not_awaited()

    async def test_unknown_selection_and_evidence_bound_are_rejected(
        self,
    ) -> None:
        with patch(
            "quantmind.mind.retrieval.Runner.run",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    final_output={"node_ids": [str(uuid4())]}
                )
            ),
        ):
            with self.assertRaisesRegex(ValueError, "unknown structure node"):
                await retrieve(
                    self.tree,
                    "Find attention.",
                    library=self.library,
                    cfg=RetrievalCfg(),
                )

        child_ids = [node.node_id for node in self.tree.walk_dfs()]
        with patch(
            "quantmind.mind.retrieval.Runner.run",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    final_output={
                        "node_ids": [str(value) for value in child_ids]
                    }
                )
            ),
        ):
            with self.assertRaisesRegex(
                RetrievalError,
                "max_evidence_nodes",
            ):
                await retrieve(
                    self.tree,
                    "Find all evidence.",
                    library=self.library,
                    cfg=RetrievalCfg(max_evidence_nodes=1),
                )

    async def test_blank_question_unbound_base_and_timeout_fail_closed(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "must not be blank"):
            await retrieve(
                self.tree,
                "   ",
                library=self.library,
                cfg=RetrievalCfg(),
            )

        unbound = StructureTree(
            root_node_id=self.tree.root_node_id,
            nodes=self.tree.nodes,
        )
        with self.assertRaisesRegex(TypeError, "must expose"):
            await retrieve(
                unbound,
                "Find attention.",
                library=self.library,
                cfg=RetrievalCfg(),
            )

        async def slow_run(*args, **kwargs):
            del args, kwargs
            await asyncio.sleep(10)

        with patch(
            "quantmind.mind.retrieval.Runner.run",
            new=AsyncMock(side_effect=slow_run),
        ):
            with self.assertRaisesRegex(RetrievalError, "timeout_seconds"):
                await retrieve(
                    self.tree,
                    "Find attention.",
                    library=self.library,
                    cfg=RetrievalCfg(timeout_seconds=0.01),
                )

    def test_structure_serialization_is_bounded_and_strips_content(
        self,
    ) -> None:
        serialized = _serialize_structure(
            self.tree,
            token_budget=256,
            seed_ids={self.node.node_id},
        )

        self.assertLessEqual(len(serialized), 256 * 4)
        self.assertNotIn("content", serialized)
        self.assertIn(str(self.node.node_id), serialized)


if __name__ == "__main__":
    unittest.main()
