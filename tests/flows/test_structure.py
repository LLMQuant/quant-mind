"""Offline tests for the config-bound ``PaperFlow.build`` structure shape."""

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from agents import ModelSettings
from openai import BadRequestError

from quantmind.configs import BaseFlowCfg, PaperStructureCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import PaperFlow, PaperStructureError
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


class PaperFlowBuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_returns_self_contained_tree_per_call(self) -> None:
        provider = _FakeStructureProvider()
        parse_spy = AsyncMock(side_effect=parse_pdf)
        cfg = PaperStructureCfg(model="litellm/anthropic/claude-test")

        flow = PaperFlow(cfg, _structure_provider=provider)
        # Mutating the caller's cfg after construction must not leak in:
        # PaperFlow binds an immutable copy.
        cfg.model = "mutated-after-bind"

        with patch("quantmind.flows.paper.parse_pdf", new=parse_spy):
            tree_one = await flow.build(LocalFilePath(path=_FIXTURE))
            tree_two = await flow.build(LocalFilePath(path=_FIXTURE))

        # Binding cfg once and calling build twice re-parses per call: no
        # hidden shared/open source state survives across calls.
        self.assertEqual(parse_spy.await_count, 2)
        self.assertEqual(len(provider.calls), 2)
        self.assertIsNot(provider.calls[0][1], provider.calls[1][1])

        # The bound (copied) cfg is used, not the mutated original.
        self.assertEqual(
            tree_one.producer.model, "litellm/anthropic/claude-test"
        )

        # The tree is self-contained: leaves carry content, internals do not.
        leaves = [
            node for node in tree_one.nodes.values() if not node.children_ids
        ]
        internals = [
            node for node in tree_one.nodes.values() if node.children_ids
        ]
        self.assertTrue(leaves)
        self.assertTrue(internals)
        self.assertTrue(all(node.content for node in leaves))
        self.assertTrue(all(node.content is None for node in internals))

        # Same input + bound cfg yields the same code-owned identity.
        self.assertEqual(tree_one.id, tree_two.id)
        self.assertEqual(
            tree_one.source_revision_id, tree_two.source_revision_id
        )

    async def test_build_propagates_structure_error(self) -> None:
        provider = _RaisingStructureProvider()
        parse_spy = AsyncMock(side_effect=parse_pdf)

        flow = PaperFlow(PaperStructureCfg(), _structure_provider=provider)
        with patch("quantmind.flows.paper.parse_pdf", new=parse_spy):
            with self.assertRaises(PaperStructureError):
                await flow.build(LocalFilePath(path=_FIXTURE))

        # Fetch + parse ran for this call before the provider raised.
        self.assertEqual(parse_spy.await_count, 1)
        self.assertEqual(provider.calls, 1)

    async def test_unwired_cfg_type_raises_not_implemented(self) -> None:
        # A cfg type that is neither PaperStructureCfg nor PaperSemanticCfg selects
        # an unwired shape: build must reject it by cfg type, before any fetch
        # or parse.
        flow = PaperFlow(BaseFlowCfg())
        with patch(
            "quantmind.flows.paper.parse_pdf",
            new=AsyncMock(side_effect=AssertionError("must not parse")),
        ):
            with self.assertRaises(NotImplementedError):
                await flow.build(LocalFilePath(path=_FIXTURE))


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    return BadRequestError(
        message, response=httpx.Response(400, request=request), body=None
    )


class AgentsStructureProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_structured_output_is_the_default_for_every_model(
        self,
    ) -> None:
        # A litellm-routed model is no longer force-downgraded: it takes the
        # SDK's native strict json_schema path first, like any other provider.
        result = build_paper_result()
        cfg = PaperStructureCfg(
            model="litellm/openrouter/deepseek/deepseek-v4-flash"
        )
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
        self.assertIsNone(agent.model_settings.extra_body)

    async def test_falls_back_to_json_object_when_json_schema_is_rejected(
        self,
    ) -> None:
        result = build_paper_result()
        cfg = PaperStructureCfg(
            model="litellm/openrouter/deepseek/deepseek-v4-flash"
        )
        run_mock = AsyncMock(
            side_effect=[
                _bad_request("This response_format type is unavailable now"),
                f"```json\n{_fixture_draft().model_dump_json()}\n```",
            ]
        )
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
        self.assertEqual(run_mock.await_count, 2)
        strict_agent = run_mock.await_args_list[0].args[0]
        self.assertIs(strict_agent.output_type, PaperStructureTreeDraft)
        self.assertIsNone(strict_agent.model_settings.extra_body)
        json_agent = run_mock.await_args_list[1].args[0]
        self.assertIsNone(json_agent.output_type)
        self.assertEqual(
            json_agent.model_settings.extra_body["response_format"],
            {"type": "json_object"},
        )
        self.assertIn("final output format", json_agent.instructions)

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
