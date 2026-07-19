"""Tests for reasoning-based retrieval configuration."""

import unittest

from pydantic import ValidationError

from quantmind.configs import RetrievalCfg


class RetrievalCfgTests(unittest.TestCase):
    def test_defaults_to_bounded_single_pass(self) -> None:
        cfg = RetrievalCfg()

        self.assertEqual(cfg.grain, "single-pass")
        self.assertGreaterEqual(cfg.structure_token_budget, 256)
        self.assertGreaterEqual(cfg.max_evidence_nodes, 1)

    def test_accepts_agentic_and_litellm_model(self) -> None:
        cfg = RetrievalCfg(
            grain="agentic",
            model="litellm/anthropic/claude-test",
        )

        self.assertEqual(cfg.grain, "agentic")
        self.assertEqual(cfg.model, "litellm/anthropic/claude-test")

    def test_rejects_unknown_grain_and_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            RetrievalCfg(grain="hybrid")  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            RetrievalCfg(backend="custom")  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
