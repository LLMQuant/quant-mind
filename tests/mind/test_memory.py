"""Tests for ``quantmind.mind.memory``."""

import json
import tempfile
import unittest

from quantmind.mind.memory import HybridMemoryEngine, can_commit_to_l3


class HybridMemoryEngineTests(unittest.TestCase):
    def test_write_l1_trace_appends_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = HybridMemoryEngine(base_path=tmpdir)
            engine.write_l1_trace("Scout", {"tool": "scan_repo", "ok": True})
            content = engine.l1_path.read_text(encoding="utf-8").strip()
            parsed = json.loads(content)
            self.assertEqual(parsed["role"], "Scout")
            self.assertTrue(parsed["payload"]["ok"])

    def test_update_l2_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = HybridMemoryEngine(base_path=tmpdir)
            current = engine.update_l2_state("step", {"status": "running"})
            self.assertEqual(current["step"]["status"], "running")
            persisted = json.loads(engine.l2_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["step"]["status"], "running")

    def test_commit_to_l3_requires_all_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = HybridMemoryEngine(base_path=tmpdir)
            accepted = engine.commit_to_l3_graph(
                {
                    "id": "artifact-1",
                    "validation_confidence": 0.9,
                    "provenance": {"source": "test"},
                },
                schema_valid=True,
                dedup_ok=True,
                contradiction_free=True,
            )
            rejected = engine.commit_to_l3_graph(
                {
                    "id": "artifact-2",
                    "validation_confidence": 0.9,
                    "provenance": {"source": "test"},
                },
                schema_valid=True,
                dedup_ok=False,
                contradiction_free=True,
            )
            self.assertTrue(accepted)
            self.assertFalse(rejected)
            lines = engine.l3_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["id"], "artifact-1")


class CanCommitToL3Tests(unittest.TestCase):
    def test_rejects_without_provenance(self) -> None:
        self.assertFalse(
            can_commit_to_l3(
                {"validation_confidence": 0.9},
                schema_valid=True,
                dedup_ok=True,
                contradiction_free=True,
            )
        )

    def test_rejects_low_confidence(self) -> None:
        self.assertFalse(
            can_commit_to_l3(
                {
                    "validation_confidence": 0.5,
                    "provenance": {"source": "test"},
                },
                schema_valid=True,
                dedup_ok=True,
                contradiction_free=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
