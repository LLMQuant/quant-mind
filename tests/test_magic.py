"""Tests for ``quantmind.magic``."""

import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from typing import Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel

from quantmind.configs import (
    NewsCollectionCfg,
    NewsWindow,
    PaperSemanticCfg,
)
from quantmind.configs.paper import ArxivIdentifier, PaperInput
from quantmind.flows import collect_news
from quantmind.magic import (
    ResolvedFlowConfig,
    _introspect_flow_signature,
    _pydantic_schema_str,
    _strip_optional,
    preview_resolve,
    resolve_magic_input,
)


class StripOptionalTests(unittest.TestCase):
    def test_optional_t(self) -> None:
        self.assertIs(_strip_optional(Optional[int]), int)

    def test_pep604_union_with_none(self) -> None:
        self.assertIs(_strip_optional(int | None), int)

    def test_plain_t_unchanged(self) -> None:
        self.assertIs(_strip_optional(int), int)

    def test_union_without_none_unchanged(self) -> None:
        anno = Union[int, str]
        self.assertEqual(_strip_optional(anno), anno)


def _news_window() -> NewsWindow:
    """A minimal valid collection window for resolver fixtures."""
    return NewsWindow(
        source="pr-newswire",
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )


class IntrospectFlowSignatureTests(unittest.TestCase):
    def test_collect_news_returns_window_input_and_cfg(self) -> None:
        input_type, cfg_type = _introspect_flow_signature(collect_news)
        self.assertIs(cfg_type, NewsCollectionCfg)
        self.assertIs(input_type, NewsWindow)

    def test_resolves_postponed_annotations(self) -> None:
        async def postponed_flow(
            input: "ArxivIdentifier",
            *,
            cfg: "PaperSemanticCfg | None" = None,
        ) -> None:
            return None

        input_type, cfg_type = _introspect_flow_signature(postponed_flow)

        self.assertIs(input_type, ArxivIdentifier)
        self.assertIs(cfg_type, PaperSemanticCfg)

    def test_missing_input_param_raises(self) -> None:
        async def bad(*, cfg: PaperSemanticCfg | None = None) -> None:
            return None

        with self.assertRaises(TypeError):
            _introspect_flow_signature(bad)

    def test_missing_cfg_param_raises(self) -> None:
        async def bad(input: ArxivIdentifier) -> None:
            return None

        with self.assertRaises(TypeError):
            _introspect_flow_signature(bad)

    def test_cfg_annotation_must_be_baseflowcfg(self) -> None:
        async def bad(input: ArxivIdentifier, *, cfg: int = 0) -> None:
            return None

        with self.assertRaises(TypeError):
            _introspect_flow_signature(bad)


class PydanticSchemaStrTests(unittest.TestCase):
    def test_basemodel_renders_json_schema(self) -> None:
        # PaperSemanticCfg embeds ModelSettings which has callable fields;
        # the renderer falls back to a fields summary in that case.
        out = _pydantic_schema_str(PaperSemanticCfg)
        parsed = json.loads(out)
        self.assertEqual(parsed.get("title"), "PaperSemanticCfg")
        self.assertIn("model", parsed["fields"])

    def test_basemodel_with_clean_schema(self) -> None:
        class Clean(BaseModel):
            x: int = 0

        out = _pydantic_schema_str(Clean)
        parsed = json.loads(out)
        # Clean schema path -> emits standard "properties" key.
        self.assertIn("properties", parsed)

    def test_annotated_union_emits_one_of(self) -> None:
        out = _pydantic_schema_str(PaperInput)
        parsed = json.loads(out)
        self.assertIn("oneOf", parsed)
        # PaperInput has 5 variants; not all need schema-rendering, but
        # the rendered list should be non-empty.
        self.assertGreater(len(parsed["oneOf"]), 0)

    def test_baseinput_subclass_directly(self) -> None:
        out = _pydantic_schema_str(ArxivIdentifier)
        parsed = json.loads(out)
        self.assertEqual(parsed["properties"]["type"]["default"], "arxiv")

    def test_fallback_for_unknown_type(self) -> None:
        out = _pydantic_schema_str(int)
        # Falls back to repr.
        self.assertEqual(out, repr(int))


class ResolveMagicInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_happy_path_returns_tuple(self) -> None:
        captured: dict[str, object] = {}

        def _capture_agent(*_a: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return MagicMock(name="agent")

        # Build a fake resolver result whose final_output is a populated
        # ResolvedFlowConfig.
        resolved = ResolvedFlowConfig[NewsWindow, NewsCollectionCfg](
            input_obj=_news_window(),
            cfg_obj=NewsCollectionCfg(retain_raw_html=True),
        )
        fake_result = MagicMock()
        fake_result.final_output = resolved
        with (
            patch("quantmind.magic.Agent", side_effect=_capture_agent),
            patch(
                "quantmind.magic.Runner.run",
                new=AsyncMock(return_value=fake_result),
            ),
        ):
            inp, cfg = await resolve_magic_input(
                "collect the last day of pr-newswire",
                target_flow=collect_news,
            )
        self.assertIs(inp, resolved.input_obj)
        self.assertIs(cfg, resolved.cfg_obj)
        # Resolver agent was given a name derived from the flow.
        self.assertEqual(captured["name"], "magic_resolver_collect_news")
        self.assertEqual(captured["model"], "gpt-5.6-luna")

    async def test_custom_resolver_instructions(self) -> None:
        captured: dict[str, object] = {}

        def _capture_agent(*_a: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return MagicMock()

        resolved = ResolvedFlowConfig[NewsWindow, NewsCollectionCfg](
            input_obj=_news_window(),
            cfg_obj=NewsCollectionCfg(),
        )
        fake_result = MagicMock()
        fake_result.final_output = resolved
        template = "FLOW={flow_name} INPUT={input_schema} CFG={cfg_schema}"
        with (
            patch("quantmind.magic.Agent", side_effect=_capture_agent),
            patch(
                "quantmind.magic.Runner.run",
                new=AsyncMock(return_value=fake_result),
            ),
        ):
            await resolve_magic_input(
                "x",
                target_flow=collect_news,
                resolver_instructions=template,
            )
        instructions = captured["instructions"]
        assert isinstance(instructions, str)
        self.assertTrue(instructions.startswith("FLOW=collect_news"))
        self.assertIn("INPUT=", instructions)
        self.assertIn("CFG=", instructions)

    async def test_custom_resolver_model_used(self) -> None:
        captured: dict[str, object] = {}

        def _capture_agent(*_a: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return MagicMock()

        resolved = ResolvedFlowConfig[NewsWindow, NewsCollectionCfg](
            input_obj=_news_window(),
            cfg_obj=NewsCollectionCfg(),
        )
        fake_result = MagicMock()
        fake_result.final_output = resolved
        with (
            patch("quantmind.magic.Agent", side_effect=_capture_agent),
            patch(
                "quantmind.magic.Runner.run",
                new=AsyncMock(return_value=fake_result),
            ),
        ):
            await resolve_magic_input(
                "x",
                target_flow=collect_news,
                resolver_model="custom-resolver-model",
            )
        self.assertEqual(captured["model"], "custom-resolver-model")


class PreviewResolveTests(unittest.IsolatedAsyncioTestCase):
    async def test_prints_and_returns_tuple(self) -> None:
        resolved = ResolvedFlowConfig[NewsWindow, NewsCollectionCfg](
            input_obj=_news_window(),
            cfg_obj=NewsCollectionCfg(),
        )
        fake_result = MagicMock()
        fake_result.final_output = resolved
        with (
            patch("quantmind.magic.Agent", return_value=MagicMock()),
            patch(
                "quantmind.magic.Runner.run",
                new=AsyncMock(return_value=fake_result),
            ),
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                inp, cfg = await preview_resolve("x", target_flow=collect_news)
        self.assertIs(inp, resolved.input_obj)
        self.assertIs(cfg, resolved.cfg_obj)
        out = buf.getvalue()
        self.assertIn("input_obj:", out)
        self.assertIn("cfg_obj:", out)
        self.assertIn("pr-newswire", out)
