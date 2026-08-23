"""Tests for source-native paper structure configuration."""

import unittest

from pydantic import ValidationError

from quantmind.configs import PaperStructureCfg


class PaperStructureCfgTests(unittest.TestCase):
    def test_defaults_are_build_specific(self) -> None:
        cfg = PaperStructureCfg()

        self.assertEqual(cfg.model, "gpt-5.6-luna")
        self.assertEqual(cfg.prompt_version, "paper-structure-v3")
        self.assertIsNone(cfg.page_text_chars)
        self.assertEqual(cfg.window_chars, 80_000)
        self.assertEqual(cfg.window_overlap_pages, 1)
        self.assertEqual(cfg.max_depth, 6)
        self.assertEqual(cfg.max_nodes, 128)

    def test_invalid_page_or_tree_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            PaperStructureCfg(page_text_chars=20)
        with self.assertRaises(ValidationError):
            PaperStructureCfg(window_chars=100)
        with self.assertRaises(ValidationError):
            PaperStructureCfg(window_overlap_pages=-1)
        with self.assertRaises(ValidationError):
            PaperStructureCfg(max_nodes=0)

    def test_optional_page_clip_accepts_explicit_bound(self) -> None:
        cfg = PaperStructureCfg(page_text_chars=2_000)

        self.assertEqual(cfg.page_text_chars, 2_000)


if __name__ == "__main__":
    unittest.main()
