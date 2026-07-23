"""Tests for the structured-output helper and its JSON-object fallback."""

import unittest
from typing import Any

import httpx
from agents import Agent, ModelSettings
from openai import BadRequestError
from pydantic import BaseModel

from quantmind.utils.structured_output import (
    json_object_instructions,
    json_object_model_settings,
    run_structured,
)


class _Draft(BaseModel):
    """Small local output type used to validate the helper."""

    value: str


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    return BadRequestError(
        message, response=httpx.Response(400, request=request), body=None
    )


class RunStructuredTests(unittest.IsolatedAsyncioTestCase):
    """The strict-first ladder: fall back only on a json_schema rejection."""

    async def test_strict_path_returns_output_without_building_fallback(
        self,
    ) -> None:
        modes: list[bool] = []

        def build_agent(json_object: bool) -> Agent[Any]:
            modes.append(json_object)
            return object()  # type: ignore[return-value]

        async def run(_agent: Agent[Any]) -> Any:
            return _Draft(value="strict")  # SDK returns a parsed model

        result = await run_structured(_Draft, build_agent=build_agent, run=run)

        self.assertEqual(result, _Draft(value="strict"))
        self.assertEqual(modes, [False])  # json-object path never built

    async def test_falls_back_to_json_object_on_response_format_rejection(
        self,
    ) -> None:
        modes: list[bool] = []
        calls: list[int] = []

        def build_agent(json_object: bool) -> Agent[Any]:
            modes.append(json_object)
            return object()  # type: ignore[return-value]

        async def run(_agent: Agent[Any]) -> Any:
            calls.append(1)
            if len(calls) == 1:
                raise _bad_request(
                    "This response_format type is unavailable now"
                )
            return '```json\n{"value": "fallback"}\n```'  # raw fenced string

        result = await run_structured(_Draft, build_agent=build_agent, run=run)

        self.assertEqual(result, _Draft(value="fallback"))
        self.assertEqual(modes, [False, True])  # strict, then json-object
        self.assertEqual(len(calls), 2)

    async def test_json_object_fallback_accepts_unfenced_json(self) -> None:
        calls: list[int] = []

        async def run(_agent: Agent[Any]) -> Any:
            calls.append(1)
            if len(calls) == 1:
                raise _bad_request("json_schema is not supported by provider")
            return '{"value": "bare"}'

        result = await run_structured(
            _Draft,
            build_agent=lambda _json: object(),
            run=run,  # type: ignore[arg-type, return-value]
        )

        self.assertEqual(result, _Draft(value="bare"))

    async def test_bad_request_unrelated_to_response_format_is_reraised(
        self,
    ) -> None:
        modes: list[bool] = []

        def build_agent(json_object: bool) -> Agent[Any]:
            modes.append(json_object)
            return object()  # type: ignore[return-value]

        async def run(_agent: Agent[Any]) -> Any:
            raise _bad_request("insufficient_quota: you exceeded your quota")

        with self.assertRaises(BadRequestError):
            await run_structured(_Draft, build_agent=build_agent, run=run)

        self.assertEqual(modes, [False])  # no json-object fallback attempted


class JsonObjectHelperTests(unittest.TestCase):
    """The pieces a call site's json-object ``build_agent`` reuses."""

    def test_settings_preserve_existing_values_and_force_json_object(
        self,
    ) -> None:
        settings = json_object_model_settings(
            ModelSettings(
                max_tokens=123,
                extra_body={"provider": {"sort": "price"}},
            )
        )

        self.assertEqual(settings.max_tokens, 123)
        self.assertEqual(settings.extra_body["provider"], {"sort": "price"})
        self.assertEqual(
            settings.extra_body["response_format"],
            {"type": "json_object"},
        )

    def test_instructions_embed_schema_and_local_validation_contract(
        self,
    ) -> None:
        instructions = json_object_instructions("Return a draft.", _Draft)

        self.assertIn("final output format", instructions)
        self.assertIn("JSON Schema", instructions)
        self.assertIn('"value"', instructions)  # the schema is embedded
