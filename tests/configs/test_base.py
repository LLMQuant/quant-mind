"""Tests for configs.base."""

import unittest

from agents import ModelSettings
from pydantic import ValidationError

from quantmind.configs.base import BaseFlowCfg, BaseInput


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


class PackageExportTests(unittest.TestCase):
    def test_top_level_imports(self):
        from quantmind.configs import (
            BaseFlowCfg as BaseFlowCfgExport,
        )
        from quantmind.configs import (
            BaseInput as BaseInputExport,
        )
        from quantmind.configs import (
            EarningsFlowCfg,
            NewsFlowCfg,
            PaperFlowCfg,
        )

        self.assertTrue(issubclass(PaperFlowCfg, BaseFlowCfgExport))
        self.assertTrue(issubclass(NewsFlowCfg, BaseFlowCfgExport))
        self.assertTrue(issubclass(EarningsFlowCfg, BaseFlowCfgExport))
        self.assertEqual(BaseInputExport.__name__, "BaseInput")


if __name__ == "__main__":
    unittest.main()
