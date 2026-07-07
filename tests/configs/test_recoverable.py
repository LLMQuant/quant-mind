"""Tests for recoverable validation helpers."""

import unittest

from pydantic import BaseModel, ConfigDict

from quantmind.configs.recoverable import RawFallbackNode, RecoverableValidation


class _StrictSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    score: float


class RecoverableValidationTests(unittest.TestCase):
    def test_returns_validated_model_on_success(self) -> None:
        validator = RecoverableValidation(_StrictSample)
        out = validator.execute_safely({"name": "node-1", "score": 0.91})
        self.assertIsInstance(out, _StrictSample)
        assert isinstance(out, _StrictSample)
        self.assertEqual(out.name, "node-1")
        self.assertEqual(out.score, 0.91)

    def test_returns_fallback_node_on_validation_error(self) -> None:
        validator = RecoverableValidation(_StrictSample)
        out = validator.execute_safely(
            {"name": "node-1", "score": 0.91, "unexpected": "x"}
        )
        self.assertIsInstance(out, RawFallbackNode)
        assert isinstance(out, RawFallbackNode)
        self.assertEqual(out.target_schema, "_StrictSample")
        self.assertTrue(out.context_loss_prevented)
        self.assertEqual(out.raw_payload["unexpected"], "x")
        self.assertIn("Extra inputs are not permitted", out.error_message)


if __name__ == "__main__":
    unittest.main()
