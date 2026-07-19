"""Offline tests for paper structure-tree construction."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from agents import ModelSettings

from quantmind.configs import PaperFlowCfg
from quantmind.flows import build_paper_structure_tree
from quantmind.flows._paper_structure import (
    PaperStructureError,
    _AgentsPaperStructureProvider,
    _structure_model_settings,
)
from quantmind.knowledge import (
    PaperStructureNodeDraft,
    PaperStructureTreeDraft,
)
from quantmind.preprocess import OutlineSignals, ParsedDocument, ParsedPage
from tests.paper_helpers import build_paper_result


def _document() -> ParsedDocument:
    result = build_paper_result()
    return ParsedDocument(
        source_hash=result.source_revision.parsed.source_hash,
        parser_name="fixture",
        parser_version="1",
        cleanup_version="1",
        pages=tuple(
            ParsedPage(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                text=page.text,
                blocks=(),
            )
            for page in result.source_revision.parsed.pages
        ),
    )


def _draft() -> PaperStructureTreeDraft:
    return PaperStructureTreeDraft(
        root=PaperStructureNodeDraft(
            title="Attention Is All You Need",
            summary="The complete paper.",
            chunk_indices=(0, 1, 2),
            children=(
                PaperStructureNodeDraft(
                    title="Architecture",
                    summary="The architecture.",
                    chunk_indices=(0,),
                ),
                PaperStructureNodeDraft(
                    title="Attention and results",
                    summary="The attention mechanism and results.",
                    chunk_indices=(1, 2),
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

    async def structure(self, signals, chunk_set, *, cfg):
        self.calls.append((signals, chunk_set, cfg))
        return _draft()


class PaperStructureBuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_uses_one_draft_and_keeps_model_identity(self) -> None:
        result = build_paper_result()
        provider = _FakeStructureProvider()
        cfg = PaperFlowCfg(model="litellm/anthropic/claude-test")

        tree = await build_paper_structure_tree(
            _document(),
            result.chunk_set,
            cfg=cfg,
            _structure_provider=provider,
        )

        self.assertEqual(len(provider.calls), 1)
        self.assertIs(provider.calls[0][1], result.chunk_set)
        self.assertEqual(tree.producer.model, cfg.model)
        self.assertEqual(tree.producer.input_chunk_set_id, result.chunk_set.id)
        self.assertEqual(len(tree.nodes), 3)
        self.assertTrue(
            all(node.content is None for node in tree.nodes.values())
        )

    async def test_chunk_page_absent_from_document_is_rejected(self) -> None:
        result = build_paper_result()
        one_page = _document()
        one_page = ParsedDocument(
            source_hash=one_page.source_hash,
            parser_name=one_page.parser_name,
            parser_version=one_page.parser_version,
            cleanup_version=one_page.cleanup_version,
            pages=one_page.pages[:1],
        )

        with self.assertRaisesRegex(ValueError, "absent from the document"):
            await build_paper_structure_tree(
                one_page,
                result.chunk_set,
                _structure_provider=_FakeStructureProvider(),
            )


class AgentsStructureProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_runs_one_structured_output_agent(self) -> None:
        result = build_paper_result()
        cfg = PaperFlowCfg(model="litellm/openrouter/test-model")
        run_mock = AsyncMock(return_value=_draft())
        with patch(
            "quantmind.flows._paper_structure.run_with_observability",
            new=run_mock,
        ):
            draft = await _AgentsPaperStructureProvider().structure(
                signals=_empty_signals(),
                chunk_set=result.chunk_set,
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
                    chunk_set=result.chunk_set,
                    cfg=PaperFlowCfg(timeout_seconds=0.01),
                )

    def test_output_token_limit_uses_the_stricter_bound(self) -> None:
        capped = _structure_model_settings(
            PaperFlowCfg(
                max_structure_output_tokens=256,
                model_settings=ModelSettings(max_tokens=1024),
            )
        )
        lower = _structure_model_settings(
            PaperFlowCfg(
                max_structure_output_tokens=256,
                model_settings=ModelSettings(max_tokens=128),
            )
        )
        self.assertEqual(capped.max_tokens, 256)
        self.assertEqual(lower.max_tokens, 128)


if __name__ == "__main__":
    unittest.main()
