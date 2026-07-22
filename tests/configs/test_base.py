"""Tests for configs.base."""

import os
import unittest
from unittest.mock import patch

from agents import ModelSettings
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import ValidationError

from quantmind.configs.base import (
    ATLASCLOUD_BASE_URL,
    ATLASCLOUD_DEFAULT_CHAT_MODEL,
    ATLASCLOUD_DEFAULT_REASONING_MODEL,
    BaseFlowCfg,
    BaseInput,
    atlascloud_model,
    is_atlascloud_model,
    resolve_agent_model,
)


class BaseFlowCfgTests(unittest.TestCase):
    def test_defaults(self):
        cfg = BaseFlowCfg()
        self.assertEqual(cfg.model, "gpt-4o")
        self.assertEqual(cfg.max_turns, 10)
        self.assertAlmostEqual(cfg.timeout_seconds, 300.0)
        self.assertIsNone(cfg.model_settings)
        self.assertIsNone(cfg.memory_dir)
        self.assertTrue(cfg.archive_trajectory)
        self.assertTrue(cfg.enable_default_guardrails)
        self.assertFalse(cfg.tracing_disabled)

    def test_extra_forbidden(self):
        with self.assertRaises(ValidationError):
            BaseFlowCfg(unknown=True)  # type: ignore[call-arg]

    def test_model_settings_accepted(self):
        ms = ModelSettings(temperature=0.1)
        cfg = BaseFlowCfg(model_settings=ms)
        assert cfg.model_settings is not None
        self.assertEqual(cfg.model_settings.temperature, 0.1)


class BaseInputTests(unittest.TestCase):
    def test_extra_forbidden(self):
        class _SampleInput(BaseInput):
            x: int

        with self.assertRaises(ValidationError):
            _SampleInput(x=1, y=2)  # type: ignore[call-arg]


class AtlasCloudModelTests(unittest.TestCase):
    def test_atlascloud_model_defaults_and_preserves_aliases(self) -> None:
        self.assertEqual(
            atlascloud_model(),
            f"atlascloud/{ATLASCLOUD_DEFAULT_CHAT_MODEL}",
        )
        self.assertEqual(
            atlascloud_model(ATLASCLOUD_DEFAULT_REASONING_MODEL),
            f"atlascloud/{ATLASCLOUD_DEFAULT_REASONING_MODEL}",
        )
        self.assertEqual(
            atlascloud_model("atlas-cloud/qwen/qwen3.5-flash"),
            "atlas-cloud/qwen/qwen3.5-flash",
        )
        self.assertTrue(is_atlascloud_model("atlas/qwen/qwen3.5-flash"))

    def test_atlascloud_model_rejects_blank_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be blank"):
            atlascloud_model(" ")

    def test_resolve_agent_model_uses_atlascloud_environment(self) -> None:
        env = {
            "ATLASCLOUD_API_KEY": "atlas-test-key",
            "ATLASCLOUD_BASE_URL": "https://atlas.test/v1/",
        }
        with patch.dict(os.environ, env, clear=False):
            model = resolve_agent_model("atlascloud/qwen/qwen3.5-flash")

        self.assertIsInstance(model, LitellmModel)
        assert isinstance(model, LitellmModel)
        self.assertEqual(model.model, "openai/qwen/qwen3.5-flash")
        self.assertEqual(model.api_key, "atlas-test-key")
        self.assertEqual(model.base_url, "https://atlas.test/v1")

    def test_resolve_agent_model_requires_api_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ATLASCLOUD_API_KEY": "",
                "ATLAS_CLOUD_API_KEY": "",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "ATLASCLOUD_API_KEY"):
                resolve_agent_model("atlascloud/qwen/qwen3.5-flash")

    def test_resolve_agent_model_leaves_non_atlas_values_unchanged(
        self,
    ) -> None:
        self.assertEqual(
            resolve_agent_model("litellm/anthropic/claude-test"),
            "litellm/anthropic/claude-test",
        )


class PackageExportTests(unittest.TestCase):
    def test_top_level_imports(self):
        from quantmind.configs import (
            ATLASCLOUD_BASE_URL as ATLASCLOUD_BASE_URL_EXPORT,
        )
        from quantmind.configs import (
            BaseFlowCfg as BaseFlowCfgExport,
        )
        from quantmind.configs import (
            BaseInput as BaseInputExport,
        )
        from quantmind.configs import (
            EarningsFlowCfg,
            NewsCollectionCfg,
            PaperFlowCfg,
        )

        self.assertTrue(issubclass(PaperFlowCfg, BaseFlowCfgExport))
        self.assertTrue(issubclass(NewsCollectionCfg, BaseFlowCfgExport))
        self.assertTrue(issubclass(EarningsFlowCfg, BaseFlowCfgExport))
        self.assertEqual(BaseInputExport.__name__, "BaseInput")
        self.assertEqual(ATLASCLOUD_BASE_URL_EXPORT, ATLASCLOUD_BASE_URL)


if __name__ == "__main__":
    unittest.main()
