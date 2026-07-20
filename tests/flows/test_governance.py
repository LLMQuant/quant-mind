"""Tests for ``quantmind.flows.governance``."""

import tempfile
import unittest
from pathlib import Path

from quantmind.flows.governance import (
    GovernancePolicy,
    GovernancePolicyError,
    enforce_l3_commit_gates,
    ensure_tool_allowed,
    load_governance_policy,
    loop_budget_manager,
    run_fallback_policy,
)


class GovernanceLoaderTests(unittest.TestCase):
    def test_load_default_policy(self) -> None:
        policy = load_governance_policy()
        self.assertEqual(policy.version, "2.0.0")
        self.assertEqual(policy.global_settings.loop_budget_max, 5)
        self.assertIn("tolerant_ingestion", policy.scenarios)

    def test_load_empty_policy_raises(self) -> None:
        # Use a temp directory instead of writing into the repo's tests/
        # tree; keeps the test safe in read-only envs and under parallel
        # test runs.
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_path = Path(tmp_dir) / "empty_governance.yaml"
            empty_path.write_text("", encoding="utf-8")
            with self.assertRaises(GovernancePolicyError):
                load_governance_policy(empty_path)


class ToolAllowlistTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_governance_policy()

    def test_allowed_tool_passes(self) -> None:
        ensure_tool_allowed(
            self.policy,
            scenario_name="architecture_analysis",
            role="Scout",
            tool_name="scan_repo",
        )

    def test_disallowed_tool_raises(self) -> None:
        with self.assertRaises(PermissionError):
            ensure_tool_allowed(
                self.policy,
                scenario_name="architecture_analysis",
                role="Scout",
                tool_name="graph_write",
            )


class LoopBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_governance_policy()

    def test_budget_consumption(self) -> None:
        manager = loop_budget_manager(self.policy)
        self.assertEqual(manager.remaining, 5)
        self.assertEqual(manager.consume(), 4)
        self.assertEqual(manager.consume(steps=2), 2)

    def test_budget_overflow_raises(self) -> None:
        manager = loop_budget_manager(self.policy)
        manager.consume(steps=5)
        with self.assertRaises(RuntimeError):
            manager.consume()


class FallbackPolicyTests(unittest.TestCase):
    def test_quarantine_fallback_returns_node(self) -> None:
        policy = load_governance_policy()
        node = run_fallback_policy(
            policy,
            target_schema="ExampleSchema",
            payload={"x": 1},
            error_message="invalid",
        )
        self.assertEqual(node.target_schema, "ExampleSchema")
        self.assertEqual(node.raw_payload["x"], 1)

    def test_fail_fast_policy_raises(self) -> None:
        policy = GovernancePolicy.model_validate(
            {
                "version": "2.0.0",
                "global_settings": {
                    "loop_budget_max": 1,
                    "fallback_policy": "fail_fast",
                },
                "scenarios": {},
            }
        )
        with self.assertRaises(RuntimeError):
            run_fallback_policy(
                policy,
                target_schema="ExampleSchema",
                payload={},
                error_message="x",
            )


class L3CommitGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_governance_policy()

    def test_accepts_when_all_gates_pass(self) -> None:
        ok = enforce_l3_commit_gates(
            self.policy,
            artifact={
                "validation_confidence": 0.9,
                "provenance": {"source": "test"},
            },
            schema_valid=True,
            dedup_ok=True,
            contradiction_free=True,
        )
        self.assertTrue(ok)

    def test_rejects_when_any_gate_fails(self) -> None:
        rejected = enforce_l3_commit_gates(
            self.policy,
            artifact={
                "validation_confidence": 0.9,
                "provenance": {"source": "test"},
            },
            schema_valid=True,
            dedup_ok=False,
            contradiction_free=True,
        )
        self.assertFalse(rejected)


if __name__ == "__main__":
    unittest.main()
