"""Tests for reasoning-based (agentic) retrieval configuration."""

import unittest

from pydantic import ValidationError

from quantmind.configs import RetrievalCfg


class RetrievalCfgTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = RetrievalCfg()

        self.assertEqual(cfg.model, "gpt-4o-mini")
        self.assertGreaterEqual(cfg.structure_token_budget, 256)
        self.assertGreaterEqual(cfg.max_evidence_nodes, 1)
        self.assertGreaterEqual(cfg.max_nodes_per_tool_call, 1)
        self.assertIsNone(cfg.extra_instruction)
        # Traversal-loop bound is inherited from BaseFlowCfg.
        self.assertGreaterEqual(cfg.max_turns, 1)
        # There is one agentic behavior and no strategy dispatch, so the old
        # ``grain`` selector and the ``mode`` field are both gone.
        self.assertFalse(hasattr(cfg, "grain"))
        self.assertFalse(hasattr(cfg, "mode"))

    def test_accepts_extra_instruction_and_litellm_model(self) -> None:
        cfg = RetrievalCfg(
            model="litellm/anthropic/claude-test",
            extra_instruction="Prefer the limitations section.",
            max_nodes_per_tool_call=4,
        )

        self.assertEqual(cfg.model, "litellm/anthropic/claude-test")
        self.assertEqual(
            cfg.extra_instruction, "Prefer the limitations section."
        )
        self.assertEqual(cfg.max_nodes_per_tool_call, 4)

    def test_rejects_unknown_and_removed_fields(self) -> None:
        with self.assertRaises(ValidationError):
            RetrievalCfg(backend="custom")  # type: ignore[call-arg]
        # ``grain`` and ``mode`` are no longer fields anywhere.
        with self.assertRaises(ValidationError):
            RetrievalCfg(grain="agentic")  # type: ignore[call-arg]
        with self.assertRaises(ValidationError):
            RetrievalCfg(mode="graph")  # type: ignore[call-arg]

    def test_bounds_enforce_floors(self) -> None:
        with self.assertRaises(ValidationError):
            RetrievalCfg(structure_token_budget=1)
        with self.assertRaises(ValidationError):
            RetrievalCfg(max_evidence_nodes=0)
        with self.assertRaises(ValidationError):
            RetrievalCfg(max_nodes_per_tool_call=0)


if __name__ == "__main__":
    unittest.main()
