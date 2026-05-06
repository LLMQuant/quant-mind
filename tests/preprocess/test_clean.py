"""Tests for preprocess.clean — sync text-cleaning helpers."""

import unittest

from quantmind.preprocess.clean import (
    collapse_whitespace,
    dedupe_lines,
    normalize_unicode,
)


class NormalizeUnicodeTests(unittest.TestCase):
    def test_empty_string_returns_empty(self):
        self.assertEqual(normalize_unicode(""), "")

    def test_ligatures_replaced(self):
        text = "ﬁnance ﬂows aﬃliated"
        self.assertEqual(normalize_unicode(text), "finance flows affiliated")

    def test_smart_quotes_to_ascii(self):
        text = "\u201chello\u201d \u2018world\u2019"
        self.assertEqual(normalize_unicode(text), "\"hello\" 'world'")

    def test_em_dash_to_hyphen(self):
        self.assertEqual(normalize_unicode("a\u2014b"), "a-b")

    def test_nfkc_collapses_fullwidth(self):
        # Fullwidth digits collapse to ASCII under NFKC.
        self.assertEqual(normalize_unicode("\uff11\uff12\uff13"), "123")

    def test_control_chars_dropped(self):
        text = "hello\x00world\x07!"
        self.assertEqual(normalize_unicode(text), "helloworld!")

    def test_newlines_and_tabs_preserved(self):
        text = "line1\nline2\tcol"
        self.assertEqual(normalize_unicode(text), "line1\nline2\tcol")


class CollapseWhitespaceTests(unittest.TestCase):
    def test_empty_string_returns_empty(self):
        self.assertEqual(collapse_whitespace(""), "")

    def test_runs_of_spaces_collapsed(self):
        self.assertEqual(collapse_whitespace("hello    world"), "hello world")

    def test_tabs_treated_as_spaces(self):
        self.assertEqual(collapse_whitespace("a\t\t\tb"), "a b")

    def test_paragraph_breaks_preserved(self):
        text = "para one\n\npara two"
        self.assertEqual(collapse_whitespace(text), "para one\n\npara two")

    def test_more_than_two_newlines_collapsed_to_two(self):
        text = "a\n\n\n\nb"
        self.assertEqual(collapse_whitespace(text), "a\n\nb")

    def test_trailing_whitespace_per_line_stripped(self):
        text = "line one   \nline two\t\t"
        self.assertEqual(collapse_whitespace(text), "line one\nline two")

    def test_leading_and_trailing_blank_lines_stripped(self):
        self.assertEqual(collapse_whitespace("\n\n  hi  \n\n"), "hi")


class DedupeLinesTests(unittest.TestCase):
    def test_empty_string_returns_empty(self):
        self.assertEqual(dedupe_lines(""), "")

    def test_consecutive_duplicates_removed(self):
        text = "header\nheader\nbody"
        self.assertEqual(dedupe_lines(text), "header\nbody")

    def test_non_consecutive_duplicates_preserved(self):
        text = "header\nbody\nheader"
        self.assertEqual(dedupe_lines(text), "header\nbody\nheader")

    def test_blank_lines_not_collapsed(self):
        text = "\n\n\n"
        self.assertEqual(dedupe_lines(text), "\n\n\n")

    def test_whitespace_insensitive_dedupe(self):
        text = "Page 3\n  Page 3  \nbody"
        self.assertEqual(dedupe_lines(text), "Page 3\nbody")
