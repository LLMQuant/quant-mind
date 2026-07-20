"""Offline tests for library-free reasoning-based retrieval.

Every test drives ``retrieve(tree, question)`` over an in-memory, self-contained
structure tree built by ``tests.paper_helpers``. No library is opened or mocked;
the only fake is the SDK ``Runner.run`` call, which stands in for the model.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from quantmind.configs import RetrievalCfg
from quantmind.knowledge import PaperStructureTree, StructureTree
from quantmind.mind import RetrievalError, RetrievalEvidence, retrieve
from quantmind.mind.retrieval import _node_text, _serialize_structure
from tests.paper_helpers import build_paper_structure_tree

_PARALLEL_PROJECTIONS = "parallel projections"


def _selection(*node_ids: object) -> SimpleNamespace:
    return SimpleNamespace(
        final_output={"node_ids": [str(value) for value in node_ids]}
    )


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tree: PaperStructureTree = build_paper_structure_tree()
        self.leaf = next(
            value
            for value in self.tree.nodes.values()
            if value.title == "Attention and results"
        )
        self.root = self.tree.nodes[self.tree.root_node_id]

    async def test_single_pass_reads_content_and_locator_from_tree(
        self,
    ) -> None:
        run_mock = AsyncMock(return_value=_selection(self.leaf.node_id))
        cfg = RetrievalCfg(model="litellm/anthropic/test-model")

        with patch("quantmind.mind.retrieval.Runner.run", new=run_mock):
            evidence = await retrieve(
                self.tree,
                "How does multi-head attention work?",
                cfg=cfg,
            )

        run_mock.assert_awaited_once()
        call = run_mock.await_args
        assert call is not None
        agent = call.args[0]
        self.assertEqual(agent.model, cfg.model)
        self.assertEqual(call.kwargs["max_turns"], 1)
        # Leaf source text is stripped from the model input.
        self.assertNotIn(_PARALLEL_PROJECTIONS, call.args[1])

        self.assertEqual(len(evidence), 1)
        item = evidence[0]
        self.assertIsInstance(item, RetrievalEvidence)
        self.assertEqual(item.title, "Attention and results")
        # Content comes straight from the tree's leaf, not any library.
        self.assertEqual(item.content, self.leaf.content)
        self.assertIn(_PARALLEL_PROJECTIONS, item.content)
        self.assertEqual({c.page for c in item.citations}, {2})
        # A PaperStructureTree is identity-bearing: locator is present.
        self.assertIsNotNone(item.locator)
        assert item.locator is not None
        self.assertEqual(item.locator.artifact_id, self.tree.id)
        self.assertEqual(item.locator.member_id, self.leaf.node_id)
        self.assertEqual(
            item.locator.source_revision_id,
            self.tree.source_revision_id,
        )

    async def test_internal_node_content_concatenates_descendant_leaves(
        self,
    ) -> None:
        run_mock = AsyncMock(return_value=_selection(self.root.node_id))

        with patch("quantmind.mind.retrieval.Runner.run", new=run_mock):
            evidence = await retrieve(self.tree, "Summarize the whole paper.")

        self.assertEqual(len(evidence), 1)
        item = evidence[0]
        self.assertIsNone(self.root.content)
        # Reading-order concatenation of both leaves' content.
        expected = _node_text(self.tree, self.root.node_id)
        self.assertEqual(item.content, expected)
        self.assertIn(_PARALLEL_PROJECTIONS, item.content)
        self.assertIn("removes recurrence", item.content)

    async def test_agentic_grain_exposes_two_tools_reading_the_tree(
        self,
    ) -> None:
        async def fake_run(agent, input_value, **kwargs):
            del input_value, kwargs
            content_tool = next(
                tool for tool in agent.tools if tool.name == "get_node_content"
            )
            payload = await content_tool.on_invoke_tool(
                MagicMock(),
                json.dumps({"node_ids": [str(self.leaf.node_id)]}),
            )
            resolved = json.loads(payload)
            self.assertEqual(resolved[0]["content"], self.leaf.content)
            self.assertIn(_PARALLEL_PROJECTIONS, payload)
            return _selection(self.leaf.node_id)

        run_mock = AsyncMock(side_effect=fake_run)
        cfg = RetrievalCfg(grain="agentic", max_turns=6)

        with patch("quantmind.mind.retrieval.Runner.run", new=run_mock):
            evidence = await retrieve(
                self.tree,
                "Find the attention evidence.",
                cfg=cfg,
            )

        call = run_mock.await_args
        assert call is not None
        agent = call.args[0]
        self.assertEqual(
            {tool.name for tool in agent.tools},
            {"get_document_structure", "get_node_content"},
        )
        self.assertEqual(call.kwargs["max_turns"], 6)
        self.assertEqual(evidence[0].content, self.leaf.content)
        assert evidence[0].locator is not None
        self.assertEqual(evidence[0].locator.member_id, self.leaf.node_id)

    async def test_non_identity_tree_returns_content_without_locator(
        self,
    ) -> None:
        # A plain StructureTree carries the same self-contained node content but
        # exposes no artifact identity, so locator is None yet content stands.
        plain = StructureTree(
            root_node_id=self.tree.root_node_id,
            nodes=self.tree.nodes,
        )
        run_mock = AsyncMock(return_value=_selection(self.leaf.node_id))

        with patch("quantmind.mind.retrieval.Runner.run", new=run_mock):
            evidence = await retrieve(plain, "How does attention work?")

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].content, self.leaf.content)
        self.assertIsNone(evidence[0].locator)

    async def test_matching_seed_is_forwarded_and_unknown_seed_is_rejected(
        self,
    ) -> None:
        run_mock = AsyncMock(return_value=_selection(self.leaf.node_id))
        with patch("quantmind.mind.retrieval.Runner.run", new=run_mock):
            await retrieve(
                self.tree,
                "Find attention.",
                seed_node_ids=[self.leaf.node_id],
            )
        call = run_mock.await_args
        assert call is not None
        payload = json.loads(call.args[1])
        self.assertEqual(payload["seed_node_ids"], [str(self.leaf.node_id)])

        with patch(
            "quantmind.mind.retrieval.Runner.run",
            new=AsyncMock(),
        ) as rejected_run:
            with self.assertRaisesRegex(
                ValueError, "address nodes in the tree"
            ):
                await retrieve(
                    self.tree,
                    "Find attention.",
                    seed_node_ids=[uuid4()],
                )
        rejected_run.assert_not_awaited()

    async def test_unknown_selection_and_evidence_bound_are_rejected(
        self,
    ) -> None:
        with patch(
            "quantmind.mind.retrieval.Runner.run",
            new=AsyncMock(return_value=_selection(uuid4())),
        ):
            with self.assertRaisesRegex(ValueError, "unknown structure node"):
                await retrieve(self.tree, "Find attention.")

        all_ids = [node.node_id for node in self.tree.walk_dfs()]
        with patch(
            "quantmind.mind.retrieval.Runner.run",
            new=AsyncMock(return_value=_selection(*all_ids)),
        ):
            with self.assertRaisesRegex(RetrievalError, "max_evidence_nodes"):
                await retrieve(
                    self.tree,
                    "Find all evidence.",
                    cfg=RetrievalCfg(max_evidence_nodes=1),
                )

    async def test_blank_question_and_timeout_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be blank"):
            await retrieve(self.tree, "   ")

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
                    cfg=RetrievalCfg(timeout_seconds=0.01),
                )

    def test_structure_serialization_is_bounded_and_strips_content(
        self,
    ) -> None:
        serialized = _serialize_structure(
            self.tree,
            token_budget=256,
            seed_ids={self.leaf.node_id},
        )

        self.assertLessEqual(len(serialized), 256 * 4)
        self.assertNotIn(_PARALLEL_PROJECTIONS, serialized)
        self.assertIn(str(self.leaf.node_id), serialized)


if __name__ == "__main__":
    unittest.main()
