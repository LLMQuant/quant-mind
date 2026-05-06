"""Tests for preprocess.fetch.arxiv — id parsing + fetch_arxiv."""

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import httpx
import respx

from quantmind.preprocess.fetch import arxiv as arxiv_mod
from quantmind.preprocess.fetch.arxiv import (
    ArxivIdParseError,
    _extract_arxiv_id,
    fetch_arxiv,
)


def _stub_arxiv_result(
    *,
    pdf_url: str = "https://arxiv.org/pdf/2401.12345v1",
    title: str = "A Test Paper",
) -> SimpleNamespace:
    return SimpleNamespace(
        pdf_url=pdf_url,
        title=title,
        authors=["Alice Smith", "Bob Jones"],
        summary="Abstract goes here.",
        published=datetime(2024, 4, 15, tzinfo=timezone.utc),
        primary_category="q-fin.ST",
        categories=["q-fin.ST", "stat.ML"],
    )


class ExtractArxivIdTests(unittest.TestCase):
    def test_modern_bare_id(self):
        self.assertEqual(_extract_arxiv_id("2401.12345"), "2401.12345")

    def test_with_version_suffix(self):
        self.assertEqual(_extract_arxiv_id("2401.12345v3"), "2401.12345v3")

    def test_arxiv_prefix(self):
        self.assertEqual(_extract_arxiv_id("arXiv:2401.12345"), "2401.12345")

    def test_lower_arxiv_prefix(self):
        self.assertEqual(_extract_arxiv_id("arxiv:2401.12345"), "2401.12345")

    def test_abs_url(self):
        self.assertEqual(
            _extract_arxiv_id("https://arxiv.org/abs/2401.12345"),
            "2401.12345",
        )

    def test_pdf_url(self):
        self.assertEqual(
            _extract_arxiv_id("https://arxiv.org/pdf/2401.12345v2.pdf"),
            "2401.12345v2",
        )

    def test_legacy_id(self):
        self.assertEqual(_extract_arxiv_id("cs.AI/0102001"), "cs.AI/0102001")

    def test_invalid_raises(self):
        with self.assertRaises(ArxivIdParseError):
            _extract_arxiv_id("not an arxiv id")


class FetchArxivTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_raw_paper(self):
        pdf_url = "https://arxiv.org/pdf/2401.12345v1"
        with mock.patch.object(
            arxiv_mod,
            "_fetch_metadata_sync",
            return_value=_stub_arxiv_result(pdf_url=pdf_url),
        ):
            with respx.mock(assert_all_called=True) as router:
                router.get(pdf_url).mock(
                    return_value=httpx.Response(
                        200, content=b"%PDF-1.4 fake bytes"
                    )
                )
                result = await fetch_arxiv("arXiv:2401.12345")

        self.assertEqual(result.arxiv_id, "2401.12345")
        self.assertEqual(result.bytes, b"%PDF-1.4 fake bytes")
        self.assertEqual(result.content_type, "application/pdf")
        self.assertEqual(result.title, "A Test Paper")
        self.assertEqual(result.authors, ("Alice Smith", "Bob Jones"))
        self.assertEqual(result.primary_category, "q-fin.ST")
        self.assertEqual(result.categories, ("q-fin.ST", "stat.ML"))
        self.assertEqual(result.source_url, pdf_url)
        assert result.published_at is not None
        self.assertEqual(result.published_at.tzinfo, timezone.utc)
        self.assertEqual(result.published_at.year, 2024)

    async def test_naive_published_promoted_to_utc(self):
        naive_published = datetime(2024, 4, 15, 12, 0)
        result_obj = _stub_arxiv_result()
        result_obj.published = naive_published
        pdf_url = result_obj.pdf_url

        with mock.patch.object(
            arxiv_mod, "_fetch_metadata_sync", return_value=result_obj
        ):
            with respx.mock(assert_all_called=True) as router:
                router.get(pdf_url).mock(
                    return_value=httpx.Response(200, content=b"%PDF")
                )
                result = await fetch_arxiv("2401.12345")

        assert result.published_at is not None
        self.assertEqual(result.published_at.tzinfo, timezone.utc)
        self.assertEqual(result.published_at.hour, 12)

    async def test_lookup_failure_propagates(self):
        with mock.patch.object(
            arxiv_mod,
            "_fetch_metadata_sync",
            side_effect=LookupError("not found"),
        ):
            with self.assertRaises(LookupError):
                await fetch_arxiv("2401.99999")

    async def test_invalid_id_short_circuits(self):
        with self.assertRaises(ArxivIdParseError):
            await fetch_arxiv("not an id")

    async def test_pdf_download_error_propagates(self):
        pdf_url = "https://arxiv.org/pdf/2401.12345v1"
        with mock.patch.object(
            arxiv_mod,
            "_fetch_metadata_sync",
            return_value=_stub_arxiv_result(pdf_url=pdf_url),
        ):
            with respx.mock(assert_all_called=True) as router:
                router.get(pdf_url).mock(return_value=httpx.Response(500))
                with self.assertRaises(httpx.HTTPStatusError):
                    await fetch_arxiv("2401.12345")
