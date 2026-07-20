"""Tests for reasoning-based retrieval configuration."""

import unittest

from pydantic import ValidationError

from quantmind.configs import AgenticRetrievalCfg, RetrievalCfg


class RetrievalCfgTests(unittest.TestCase):
    def test_base_cfg_holds_shared_bounds_without_a_strategy_field(
        self,
    ) -> None:
        cfg = RetrievalCfg()

        self.assertEqual(cfg.model, "gpt-4o-mini")
        self.assertGreaterEqual(cfg.structure_token_budget, 256)
        self.assertGreaterEqual(cfg.max_evidence_nodes, 1)
        # The old single-pass/agentic ``grain`` selector is gone: strategy is
        # picked by the cfg *type*, not a field on the base cfg.
        self.assertFalse(hasattr(cfg, "grain"))

    def test_agentic_cfg_defaults_to_tree_mode(self) -> None:
        cfg = AgenticRetrievalCfg()

        self.assertIsInstance(cfg, RetrievalCfg)
        self.assertEqual(cfg.mode, "tree")
        self.assertIsNone(cfg.extra_instruction)
        self.assertGreaterEqual(cfg.max_nodes_per_tool_call, 1)
        # Traversal-loop bound is inherited from BaseFlowCfg.
        self.assertGreaterEqual(cfg.max_turns, 1)

    def test_agentic_cfg_accepts_extra_instruction_and_litellm_model(
        self,
    ) -> None:
        cfg = AgenticRetrievalCfg(
            model="litellm/anthropic/claude-test",
            extra_instruction="Prefer the limitations section.",
            max_nodes_per_tool_call=4,
        )

        self.assertEqual(cfg.model, "litellm/anthropic/claude-test")
        self.assertEqual(
            cfg.extra_instruction, "Prefer the limitations section."
        )
        self.assertEqual(cfg.max_nodes_per_tool_call, 4)

    def test_rejects_unknown_mode_and_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            AgenticRetrievalCfg(mode="graph")  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            AgenticRetrievalCfg(backend="custom")  # type: ignore[call-arg]
        # ``grain`` is no longer a field anywhere.
        with self.assertRaises(ValidationError):
            RetrievalCfg(grain="agentic")  # type: ignore[call-arg]

    def test_bounds_enforce_floors(self) -> None:
        with self.assertRaises(ValidationError):
            RetrievalCfg(structure_token_budget=1)
        with self.assertRaises(ValidationError):
            RetrievalCfg(max_evidence_nodes=0)
        with self.assertRaises(ValidationError):
            AgenticRetrievalCfg(max_nodes_per_tool_call=0)


if __name__ == "__main__":
    unittest.main()
