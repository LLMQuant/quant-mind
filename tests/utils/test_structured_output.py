"""Tests for JSON-object structured-output compatibility helpers."""

import unittest

from agents import ModelSettings
from pydantic import BaseModel

from quantmind.utils.structured_output import (
    json_object_instructions,
    json_object_model_settings,
    requires_json_object_mode,
    validate_json_object,
)


class _Draft(BaseModel):
    """Small local output type used to validate JSON mode."""

    value: str


class StructuredOutputTests(unittest.TestCase):
    def test_litellm_routes_use_json_object_mode(self) -> None:
        self.assertTrue(
            requires_json_object_mode(
                "litellm/openrouter/deepseek/deepseek-v4-flash"
            )
        )
        self.assertFalse(requires_json_object_mode("gpt-5.6-luna"))

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

    def test_instructions_and_validation_accept_a_fenced_json_object(
        self,
    ) -> None:
        instructions = json_object_instructions("Return a draft.", _Draft)
        draft = validate_json_object('```json\n{"value": "ok"}\n```', _Draft)

        self.assertIn("final output format", instructions)
        self.assertIn("JSON Schema", instructions)
        self.assertEqual(draft, _Draft(value="ok"))
