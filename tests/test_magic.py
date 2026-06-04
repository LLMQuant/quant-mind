"""Tests for ``quantmind.magic``."""

import io
import json
import unittest
from contextlib import redirect_stdout
from typing import Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

from agents import AgentOutputSchema
from pydantic import BaseModel

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import ArxivIdentifier, PaperInput
from quantmind.flows import paper_flow
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


class IntrospectFlowSignatureTests(unittest.TestCase):
    def test_paper_flow_returns_paper_input_and_cfg(self) -> None:
        input_type, cfg_type = _introspect_flow_signature(paper_flow)
        self.assertIs(cfg_type, PaperFlowCfg)
        # PaperInput is the Annotated[Union[...]] alias; pass through.
        self.assertEqual(input_type, PaperInput)

    def test_missing_input_param_raises(self) -> None:
        async def bad(*, cfg: PaperFlowCfg | None = None) -> None:
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
        # PaperFlowCfg embeds ModelSettings, but model_settings is
        # SkipJsonSchema'd, so the cfg now renders a FULL JSON schema (no
        # fields-summary fallback) with model_settings omitted.
        out = _pydantic_schema_str(PaperFlowCfg)
        parsed = json.loads(out)
        self.assertEqual(parsed.get("title"), "PaperFlowCfg")
        self.assertIn("model", parsed["properties"])
        self.assertNotIn("model_settings", parsed["properties"])

    def test_unschemable_field_falls_back_to_fields_summary(self) -> None:
        # Defensive fallback in _pydantic_schema_str: a model with a genuinely
        # un-schemable field still yields a usable name+fields summary.
        from collections.abc import Callable

        from pydantic import ConfigDict

        class _Unschemable(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            fn: Callable[[], int]

        out = _pydantic_schema_str(_Unschemable)
        parsed = json.loads(out)
        self.assertEqual(parsed["title"], "_Unschemable")
        self.assertIn("fn", parsed["fields"])

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
        resolved = ResolvedFlowConfig[PaperInput, PaperFlowCfg](
            input_obj=ArxivIdentifier(id="2604.12345"),
            cfg_obj=PaperFlowCfg(model="gpt-test"),
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
                "fetch arxiv 2604.12345 about momentum",
                target_flow=paper_flow,
            )
        self.assertIs(inp, resolved.input_obj)
        self.assertIs(cfg, resolved.cfg_obj)
        # Resolver agent was given a name derived from the flow.
        self.assertEqual(captured["name"], "magic_resolver_paper_flow")
        self.assertEqual(captured["model"], "gpt-4o-mini")
        # Output type is wrapped non-strict so the union/knowledge schema is
        # accepted (guards the AgentOutputSchema wrap in resolve_magic_input).
        self.assertIsInstance(captured["output_type"], AgentOutputSchema)
        self.assertFalse(captured["output_type"].is_strict_json_schema())

    async def test_custom_resolver_instructions(self) -> None:
        captured: dict[str, object] = {}

        def _capture_agent(*_a: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return MagicMock()

        resolved = ResolvedFlowConfig[PaperInput, PaperFlowCfg](
            input_obj=ArxivIdentifier(id="x"),
            cfg_obj=PaperFlowCfg(),
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
                target_flow=paper_flow,
                resolver_instructions=template,
            )
        instructions = captured["instructions"]
        assert isinstance(instructions, str)
        self.assertTrue(instructions.startswith("FLOW=paper_flow"))
        self.assertIn("INPUT=", instructions)
        self.assertIn("CFG=", instructions)

    async def test_custom_resolver_model_used(self) -> None:
        captured: dict[str, object] = {}

        def _capture_agent(*_a: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return MagicMock()

        resolved = ResolvedFlowConfig[PaperInput, PaperFlowCfg](
            input_obj=ArxivIdentifier(id="x"),
            cfg_obj=PaperFlowCfg(),
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
                target_flow=paper_flow,
                resolver_model="claude-3-5-sonnet",
            )
        self.assertEqual(captured["model"], "claude-3-5-sonnet")


class PreviewResolveTests(unittest.IsolatedAsyncioTestCase):
    async def test_prints_and_returns_tuple(self) -> None:
        resolved = ResolvedFlowConfig[PaperInput, PaperFlowCfg](
            input_obj=ArxivIdentifier(id="2604.12345"),
            cfg_obj=PaperFlowCfg(),
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
                inp, cfg = await preview_resolve("x", target_flow=paper_flow)
        self.assertIs(inp, resolved.input_obj)
        self.assertIs(cfg, resolved.cfg_obj)
        out = buf.getvalue()
        self.assertIn("input_obj:", out)
        self.assertIn("cfg_obj:", out)
        self.assertIn("2604.12345", out)


class ResolverOutputSchemaTests(unittest.TestCase):
    def test_resolver_output_type_builds_despite_model_settings(self) -> None:
        # The real resolver output-type construction (see resolve_magic_input).
        # Two layers had to be fixed and both are exercised here: SkipJsonSchema
        # on model_settings lets the schema build at all (no CallableSchema), and
        # strict_json_schema=False accepts the additionalProperties the
        # discriminated-union + knowledge models emit. Either layer alone
        # crashed — caught by the live-fire, not the mocked-Runner tests above.
        schema_obj = AgentOutputSchema(
            ResolvedFlowConfig[PaperInput, PaperFlowCfg],
            strict_json_schema=False,
        )
        js = schema_obj.json_schema()
        self.assertIsInstance(js, dict)
        # model_settings is intentionally absent from the LLM-facing schema.
        self.assertNotIn("model_settings", json.dumps(js))
