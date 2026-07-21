"""Tests for deterministic outline signals."""

import unittest
from pathlib import Path

from quantmind.preprocess import (
    BoundingBox,
    ParsedDocument,
    ParsedPage,
    TextBlock,
    extract_outline_signals,
    parse_pdf,
)

_GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "paper"
    / "golden"
    / "paper.pdf"
)


def _block(
    text: str,
    page_number: int,
    *,
    y0: float,
    font_size: float,
) -> TextBlock:
    return TextBlock(
        text=text,
        page_number=page_number,
        bbox=BoundingBox(x0=40, y0=y0, x1=560, y1=y0 + 16),
        font_size=font_size,
    )


def _page(
    page_number: int,
    text: str,
    blocks: tuple[TextBlock, ...],
) -> ParsedPage:
    return ParsedPage(
        page_number=page_number,
        width=612,
        height=792,
        text=text,
        blocks=blocks,
    )


class OutlineSignalsTests(unittest.TestCase):
    def test_detects_toc_headings_and_printed_page_reset(self) -> None:
        document = ParsedDocument(
            source_hash="a" * 64,
            parser_name="fixture",
            parser_version="1",
            cleanup_version="1",
            pages=(
                _page(
                    1,
                    "Contents\n1 Introduction ........ 1\n2 Method ........ 2",
                    (
                        _block("REPORT HEADER", 1, y0=20, font_size=8),
                        _block("Contents", 1, y0=80, font_size=18),
                    ),
                ),
                _page(
                    2,
                    "Preface",
                    (
                        _block("REPORT HEADER", 2, y0=20, font_size=8),
                        _block("Preface", 2, y0=100, font_size=16),
                    ),
                ),
                _page(
                    3,
                    "1 Introduction\nSee Section 2 for the method.\n1",
                    (
                        _block("REPORT HEADER", 3, y0=20, font_size=8),
                        _block("1 Introduction", 3, y0=90, font_size=16),
                        _block(
                            "See Section 2 for the method.",
                            3,
                            y0=130,
                            font_size=10,
                        ),
                        _block("1", 3, y0=740, font_size=9),
                    ),
                ),
                _page(
                    4,
                    "2 Method\nBody text for the method section.\n2",
                    (
                        _block("REPORT HEADER", 4, y0=20, font_size=8),
                        _block("2 Method", 4, y0=90, font_size=16),
                        _block(
                            "Body text for the method section.",
                            4,
                            y0=130,
                            font_size=10,
                        ),
                        _block("2", 4, y0=740, font_size=9),
                    ),
                ),
            ),
        )

        signals = extract_outline_signals(document)

        self.assertEqual(signals.table_of_contents_pages, (1,))
        self.assertEqual(signals.printed_page_offset, 2)
        self.assertIn(
            "1 Introduction", [item.text for item in signals.headings]
        )
        self.assertIn("2 Method", [item.text for item in signals.headings])
        self.assertNotIn(
            "See Section 2 for the method.",
            [item.text for item in signals.headings],
        )
        self.assertNotIn(
            "REPORT HEADER",
            [item.text for item in signals.headings],
        )

    def test_missing_toc_and_page_numbers_return_empty_hints(self) -> None:
        document = ParsedDocument(
            source_hash="b" * 64,
            parser_name="fixture",
            parser_version="1",
            cleanup_version="1",
            pages=(
                _page(
                    1,
                    "Ordinary prose without an outline.",
                    (
                        _block(
                            "Ordinary prose without an outline.",
                            1,
                            y0=100,
                            font_size=10,
                        ),
                    ),
                ),
            ),
        )

        signals = extract_outline_signals(document)

        self.assertEqual(signals.table_of_contents_pages, ())
        self.assertEqual(signals.headings, ())
        self.assertIsNone(signals.printed_page_offset)


class FixedPdfOutlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_golden_pdf_emits_ordered_physical_page_headings(
        self,
    ) -> None:
        document = await parse_pdf(_GOLDEN.read_bytes())

        signals = extract_outline_signals(document)

        numbered = [
            (heading.page_number, heading.text)
            for heading in signals.headings
            if heading.text[0].isdigit()
        ]
        self.assertIn((1, "1 Introduction"), numbered)
        self.assertIn((2, "2 Method"), numbered)
        self.assertIn((3, "2.1 Portfolio Construction"), numbered)
        self.assertIn((4, "3 Limitations"), numbered)
        self.assertEqual(signals.table_of_contents_pages, ())
        self.assertIsNone(signals.printed_page_offset)


if __name__ == "__main__":
    unittest.main()
