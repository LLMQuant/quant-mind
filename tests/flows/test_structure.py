"""Offline tests for the ``PaperFlow`` structure pipeline."""

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agents import ModelSettings

from quantmind.configs import PaperFlowCfg, PaperStructureCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import PaperFlow, PaperStructureError
from quantmind.flows._paper_summary import (
    PaperSummaryCitationDraft,
    PaperSummaryDraft,
)
from quantmind.flows.paper._structure import (
    _AgentsPaperStructureProvider,
    _structure_model_settings,
)
from quantmind.knowledge import (
    PaperStructureNodeDraft,
    PaperStructureTreeDraft,
)
from quantmind.preprocess import OutlineSignals
from quantmind.preprocess.format import parse_pdf
from tests.paper_helpers import build_paper_result

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "paper"
    / "golden"
    / "paper.pdf"
)


def _fixture_draft() -> PaperStructureTreeDraft:
    """A hierarchy over the four-page golden fixture (root covers all pages)."""
    return PaperStructureTreeDraft(
        root=PaperStructureNodeDraft(
            title="Attention Is All You Need",
            summary="The complete paper.",
            start_page=1,
            end_page=4,
            children=(
                PaperStructureNodeDraft(
                    title="Front matter",
                    summary="Abstract and introduction.",
                    start_page=1,
                    end_page=1,
                ),
                PaperStructureNodeDraft(
                    title="Body",
                    summary="Method and results.",
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
        return _fixture_draft()


class _RaisingStructureProvider:
    """Stand in for a structure draft that exceeds its runtime boundary."""

    def __init__(self) -> None:
        self.calls = 0

    async def structure(self, signals, source, *, cfg):
        del signals, source, cfg
        self.calls += 1
        raise PaperStructureError(
            "paper structure build exceeded timeout_seconds"
        )


class _FakeSummaryProvider:
    def __init__(self) -> None:
        self.calls = []

    async def summarize(self, source, chunk_set, *, cfg):
        self.calls.append((source, chunk_set, cfg))
        selected = []
        pages: set[int] = set()
        for chunk in chunk_set.chunks:
            page = chunk.source_spans[0].page_number
            if page not in pages or len(selected) < cfg.min_summary_citations:
                selected.append((chunk, page))
                pages.add(page)
            if (
                len(selected) >= cfg.min_summary_citations
                and len(pages) >= cfg.min_summary_pages
            ):
                break
        return PaperSummaryDraft(
            summary="A deterministic offline summary across physical pages.",
            citations=tuple(
                PaperSummaryCitationDraft(
                    chunk_index=chunk.position,
                    page_number=page,
                )
                for chunk, page in selected
            ),
        )


class PaperFlowStructureTests(unittest.IsolatedAsyncioTestCase):
    async def test_open_parses_once_and_pipelines_reuse_source(self) -> None:
        structure_provider = _FakeStructureProvider()
        summary_provider = _FakeSummaryProvider()
        parse_spy = AsyncMock(side_effect=parse_pdf)

        with patch("quantmind.flows.paper.parse_pdf", new=parse_spy):
            paper = await PaperFlow.open(
                LocalFilePath(path=_FIXTURE),
                _structure_provider=structure_provider,
                _summary_provider=summary_provider,
            )
            self.assertEqual(parse_spy.await_count, 1)

            tree = await paper.build_structure(
                cfg=PaperStructureCfg(model="litellm/anthropic/claude-test")
            )
            result = await paper.extract_knowledge(
                cfg=PaperFlowCfg(chunk_size=256, chunk_overlap=32)
            )
            # No pipeline re-parses: the source is bound once at open().
            self.assertEqual(parse_spy.await_count, 1)

        # Both pipelines saw the one immutable source revision.
        self.assertEqual(len(structure_provider.calls), 1)
        self.assertIs(structure_provider.calls[0][1], paper.source)
        self.assertEqual(len(summary_provider.calls), 1)
        self.assertIs(summary_provider.calls[0][0], paper.source)
        self.assertIs(result.source_revision, paper.source)

        # The tree is self-contained: leaves carry content, internals do not.
        self.assertEqual(tree.source_revision_id, paper.source.id)
        self.assertEqual(tree.producer.model, "litellm/anthropic/claude-test")
        leaves = [node for node in tree.nodes.values() if not node.children_ids]
        internals = [node for node in tree.nodes.values() if node.children_ids]
        self.assertTrue(leaves)
        self.assertTrue(all(node.content for node in leaves))
        self.assertTrue(all(node.content is None for node in internals))

    async def test_build_structure_propagates_structure_error(self) -> None:
        provider = _RaisingStructureProvider()
        parse_spy = AsyncMock(side_effect=parse_pdf)

        with patch("quantmind.flows.paper.parse_pdf", new=parse_spy):
            paper = await PaperFlow.open(
                LocalFilePath(path=_FIXTURE),
                _structure_provider=provider,
            )
            with self.assertRaises(PaperStructureError):
                await paper.build_structure()

        self.assertEqual(parse_spy.await_count, 1)
        self.assertEqual(provider.calls, 1)


class AgentsStructureProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_runs_one_structured_output_agent(self) -> None:
        result = build_paper_result()
        cfg = PaperStructureCfg(model="litellm/openrouter/test-model")
        run_mock = AsyncMock(return_value=_fixture_draft())
        with patch(
            "quantmind.flows.paper._structure.run_with_observability",
            new=run_mock,
        ):
            draft = await _AgentsPaperStructureProvider().structure(
                signals=_empty_signals(),
                source=result.source_revision,
                cfg=cfg,
            )

        self.assertEqual(draft, _fixture_draft())
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
            "quantmind.flows.paper._structure.run_with_observability",
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
