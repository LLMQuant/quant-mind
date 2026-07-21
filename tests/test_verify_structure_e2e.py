"""Regression tests for the structure-retrieval live E2E contract."""

import unittest

from scripts.verify_structure_e2e import _MODELS, _passed


class VerifyStructureE2ETests(unittest.TestCase):
    def test_e2e_exercises_the_required_openrouter_models(self) -> None:
        self.assertEqual(
            _MODELS,
            (
                "litellm/openrouter/deepseek/deepseek-v4-flash",
                "litellm/openrouter/openai/gpt-5.6-luna",
            ),
        )

    def test_passed_requires_a_built_tree_and_resolved_evidence(self) -> None:
        snapshot = {
            "node_count": 2,
            "root_page_span": [1],
            "leaf_content_count": 1,
            "evidence_titles": ["Method"],
            "evidence_all_have_content": True,
        }

        self.assertTrue(_passed(snapshot))
        snapshot["evidence_all_have_content"] = False
        self.assertFalse(_passed(snapshot))
