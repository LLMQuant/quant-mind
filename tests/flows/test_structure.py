"""Offline tests for paper structure-tree construction."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from agents import ModelSettings

from quantmind.configs import PaperStructureCfg
from quantmind.flows import PaperStructureBuilder
from quantmind.flows._paper_structure import (
    PaperStructureError,
    _AgentsPaperStructureProvider,
    _structure_model_settings,
)
from quantmind.knowledge import (
    PaperStructureNodeDraft,
    PaperStructureTreeDraft,
)
from quantmind.preprocess import OutlineSignals
from tests.paper_helpers import build_paper_result


def _draft() -> PaperStructureTreeDraft:
    return PaperStructureTreeDraft(
        root=PaperStructureNodeDraft(
            title="Attention Is All You Need",
            summary="The complete paper.",
            start_page=1,
            end_page=2,
            children=(
                PaperStructureNodeDraft(
                    title="Architecture",
                    summary="The architecture.",
                    start_page=1,
                    end_page=1,
                ),
                PaperStructureNodeDraft(
                    title="Attention and results",
                    summary="The attention mechanism and results.",
                    start_page=2,
                    end_page=2,
                ),
            ),
        )
    )


def _empty_signals() -> OutlineSignals:
    return OutlineSignals(
        table_of_contents_pages=(),
        headings=(),
        printed_page_offset=None,
    )


class _FakeStructureProvider:
    def __init__(self) -> None:
        self.calls = []

    async def structure(self, signals, source, *, cfg):
        self.calls.append((signals, source, cfg))
        return _draft()


class PaperStructureBuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_uses_one_draft_and_keeps_model_identity(self) -> None:
        result = build_paper_result()
        provider = _FakeStructureProvider()
        cfg = PaperStructureCfg(model="litellm/anthropic/claude-test")

        tree = await PaperStructureBuilder(
            cfg,
            _structure_provider=provider,
        ).build(result.source_revision)

        self.assertEqual(len(provider.calls), 1)
        self.assertIs(provider.calls[0][1], result.source_revision)
        self.assertEqual(tree.producer.model, cfg.model)
        self.assertEqual(tree.source_revision_id, result.source_revision.id)
        self.assertEqual(len(tree.nodes), 3)
        self.assertTrue(
            all(node.content is None for node in tree.nodes.values())
        )

    async def test_tree_identity_is_independent_of_chunk_configuration(
        self,
    ) -> None:
        first = build_paper_result(chunk_size=64)
        second = build_paper_result(chunk_size=256)
        builder = PaperStructureBuilder(
            _structure_provider=_FakeStructureProvider()
        )

        first_tree = await builder.build(first.source_revision)
        second_tree = await builder.build(second.source_revision)

        self.assertNotEqual(first.chunk_set.id, second.chunk_set.id)
        self.assertEqual(first_tree.id, second_tree.id)
        self.assertEqual(first_tree.nodes, second_tree.nodes)

    async def test_builder_copies_reusable_configuration(self) -> None:
        result = build_paper_result()
        cfg = PaperStructureCfg(model="stable-model")
        builder = PaperStructureBuilder(
            cfg,
            _structure_provider=_FakeStructureProvider(),
        )
        cfg.model = "mutated-model"

        tree = await builder.build(result.source_revision)

        self.assertEqual(tree.producer.model, "stable-model")


class AgentsStructureProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_runs_one_structured_output_agent(self) -> None:
        result = build_paper_result()
        cfg = PaperStructureCfg(model="litellm/openrouter/test-model")
        run_mock = AsyncMock(return_value=_draft())
        with patch(
            "quantmind.flows._paper_structure.run_with_observability",
            new=run_mock,
        ):
            draft = await _AgentsPaperStructureProvider().structure(
                signals=_empty_signals(),
                source=result.source_revision,
                cfg=cfg,
            )

        self.assertEqual(draft, _draft())
        run_mock.assert_awaited_once()
        agent = run_mock.await_args.args[0]
        self.assertEqual(agent.model, cfg.model)
        self.assertIs(agent.output_type, PaperStructureTreeDraft)

    async def test_provider_timeout_is_typed(self) -> None:
        result = build_paper_result()

        async def slow_run(*args, **kwargs):
            del args, kwargs
            await asyncio.sleep(10)

        with patch(
            "quantmind.flows._paper_structure.run_with_observability",
            new=AsyncMock(side_effect=slow_run),
        ):
            with self.assertRaises(PaperStructureError):
                await _AgentsPaperStructureProvider().structure(
                    signals=_empty_signals(),
                    source=result.source_revision,
                    cfg=PaperStructureCfg(timeout_seconds=0.01),
                )

    def test_output_token_limit_uses_the_stricter_bound(self) -> None:
        capped = _structure_model_settings(
            PaperStructureCfg(
                max_output_tokens=256,
                model_settings=ModelSettings(max_tokens=1024),
            )
        )
        lower = _structure_model_settings(
            PaperStructureCfg(
                max_output_tokens=256,
                model_settings=ModelSettings(max_tokens=128),
            )
        )
        self.assertEqual(capped.max_tokens, 256)
        self.assertEqual(lower.max_tokens, 128)


if __name__ == "__main__":
    unittest.main()
