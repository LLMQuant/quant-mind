"""Tests for ``quantmind.flows._providers``."""

import os
import unittest
from unittest.mock import patch

from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel

from quantmind.configs import PaperFlowCfg
from quantmind.flows._providers import _get_client, _resolve, configure_provider


class ResolveTests(unittest.TestCase):
    def test_gpt_prefix_resolves_to_openai_responses(self) -> None:
        provider = _resolve("gpt-4o")
        self.assertEqual(provider.name, "openai")
        self.assertIsNone(provider.base_url)
        self.assertFalse(provider.use_chat_completions)
        self.assertTrue(provider.tracing_supported)

    def test_deepseek_prefix_resolves_to_chat_completions(self) -> None:
        provider = _resolve("deepseek-chat")
        self.assertEqual(provider.name, "deepseek")
        self.assertEqual(provider.base_url, "https://api.deepseek.com/v1")
        self.assertTrue(provider.use_chat_completions)
        self.assertFalse(provider.tracing_supported)

    def test_o1_prefix_resolves_to_openai_responses(self) -> None:
        provider = _resolve("o1-mini")
        self.assertEqual(provider.name, "openai")
        self.assertFalse(provider.use_chat_completions)

    def test_o3_prefix_resolves_to_openai_responses(self) -> None:
        provider = _resolve("o3-mini")
        self.assertEqual(provider.name, "openai")

    def test_unknown_model_falls_back_to_default_openai(self) -> None:
        provider = _resolve("some-future-model")
        # Falls back to OpenAI gpt-* row.
        self.assertEqual(provider.name, "openai")

    def test_resolution_is_case_insensitive(self) -> None:
        self.assertEqual(_resolve("DEEPSEEK-Chat").name, "deepseek")
        self.assertEqual(_resolve("GPT-4o").name, "openai")


class ConfigureProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        # Clear the AsyncOpenAI cache so each test starts clean.
        _get_client.cache_clear()

    def test_openai_returns_responses_model_with_unchanged_tracing(
        self,
    ) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            cfg = PaperFlowCfg(model="gpt-4o", tracing_disabled=False)
            model_obj, effective = configure_provider(cfg)
        self.assertIsInstance(model_obj, OpenAIResponsesModel)
        self.assertFalse(effective.tracing_disabled)

    def test_deepseek_returns_chat_completions_with_tracing_forced_off(
        self,
    ) -> None:
        with patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "sk-deepseek"}, clear=False
        ):
            cfg = PaperFlowCfg(model="deepseek-chat", tracing_disabled=False)
            model_obj, effective = configure_provider(cfg)
        self.assertIsInstance(model_obj, OpenAIChatCompletionsModel)
        self.assertTrue(effective.tracing_disabled)

    def test_user_tracing_disabled_preserved_for_openai(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            cfg = PaperFlowCfg(model="gpt-4o", tracing_disabled=True)
            _, effective = configure_provider(cfg)
        self.assertTrue(effective.tracing_disabled)

    def test_missing_api_key_raises_with_clear_message(self) -> None:
        # Wipe both env vars to make sure neither leaks into the test.
        with patch.dict(
            os.environ,
            {},
            clear=False,
        ):
            os.environ.pop("DEEPSEEK_API_KEY", None)
            cfg = PaperFlowCfg(model="deepseek-chat")
            with self.assertRaises(RuntimeError) as ctx:
                configure_provider(cfg)
        msg = str(ctx.exception)
        self.assertIn("DEEPSEEK_API_KEY", msg)
        self.assertIn("deepseek", msg)

    def test_client_is_cached_per_base_url_and_api_key(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            cfg = PaperFlowCfg(model="gpt-4o")
            m1, _ = configure_provider(cfg)
            m2, _ = configure_provider(cfg)
        # Both Model wrappers should share the same underlying client.
        self.assertIs(m1._client, m2._client)
