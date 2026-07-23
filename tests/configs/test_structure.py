"""Tests for source-native paper structure configuration."""

import unittest

from pydantic import ValidationError

from quantmind.configs import PaperStructureCfg


class PaperStructureCfgTests(unittest.TestCase):
    def test_defaults_are_build_specific(self) -> None:
        cfg = PaperStructureCfg()

        self.assertEqual(cfg.model, "gpt-5.6-luna")
        self.assertEqual(cfg.prompt_version, "paper-structure-v2")
        self.assertEqual(cfg.page_text_chars, 1_200)
        self.assertEqual(cfg.max_depth, 6)
        self.assertEqual(cfg.max_nodes, 128)

    def test_invalid_page_or_tree_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            PaperStructureCfg(page_text_chars=20)
        with self.assertRaises(ValidationError):
            PaperStructureCfg(max_nodes=0)


if __name__ == "__main__":
    unittest.main()
