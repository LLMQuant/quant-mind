"""Tests for preprocess.fetch.doi — resolve_doi via Crossref."""

import unittest
from datetime import timezone

import httpx
import respx

from quantmind.preprocess.fetch.doi import (
    CROSSREF_BASE_URL,
    CrossrefMetadata,
    resolve_doi,
)


def _crossref_payload(**overrides):
    base = {
        "status": "ok",
        "message": {
            "DOI": "10.1234/example.2024.001",
            "URL": "https://example.com/paper",
            "publisher": "Example Press",
            "title": ["A Worked Example of DOI Resolution"],
            "container-title": ["Journal of Synthetic Tests"],
            "issued": {"date-parts": [[2024, 4, 15]]},
            "author": [
                {"given": "Alice", "family": "Smith"},
                {"given": "Bob", "family": "Jones"},
            ],
        },
    }
    base["message"].update(overrides)
    return base


class ResolveDoiTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_metadata_mapped(self):
        doi = "10.1234/example.2024.001"
        with respx.mock(assert_all_called=True) as router:
            router.get(f"{CROSSREF_BASE_URL}/{doi}").mock(
                return_value=httpx.Response(200, json=_crossref_payload())
            )
            result = await resolve_doi(doi)

        self.assertIsInstance(result, CrossrefMetadata)
        self.assertEqual(result.doi, doi)
        self.assertEqual(result.title, "A Worked Example of DOI Resolution")
        self.assertEqual(result.authors, ("Alice Smith", "Bob Jones"))
        self.assertEqual(result.journal, "Journal of Synthetic Tests")
        self.assertEqual(result.publisher, "Example Press")
        self.assertEqual(result.primary_url, "https://example.com/paper")
        self.assertIsNotNone(result.published_at)
        assert result.published_at is not None
        self.assertEqual(result.published_at.year, 2024)
        self.assertEqual(result.published_at.month, 4)
        self.assertEqual(result.published_at.tzinfo, timezone.utc)

    async def test_strips_https_prefix(self):
        doi = "10.1234/example.2024.001"
        with respx.mock(assert_all_called=True) as router:
            router.get(f"{CROSSREF_BASE_URL}/{doi}").mock(
                return_value=httpx.Response(200, json=_crossref_payload())
            )
            result = await resolve_doi(f"https://doi.org/{doi}")
        self.assertEqual(result.doi, doi)

    async def test_strips_doi_prefix(self):
        doi = "10.1234/example.2024.001"
        with respx.mock(assert_all_called=True) as router:
            router.get(f"{CROSSREF_BASE_URL}/{doi}").mock(
                return_value=httpx.Response(200, json=_crossref_payload())
            )
            result = await resolve_doi(f"doi:{doi}")
        self.assertEqual(result.doi, doi)

    async def test_year_only_date(self):
        doi = "10.1234/example.2024.001"
        with respx.mock(assert_all_called=True) as router:
            router.get(f"{CROSSREF_BASE_URL}/{doi}").mock(
                return_value=httpx.Response(
                    200,
                    json=_crossref_payload(issued={"date-parts": [[2024]]}),
                )
            )
            result = await resolve_doi(doi)
        assert result.published_at is not None
        self.assertEqual(result.published_at.month, 1)
        self.assertEqual(result.published_at.day, 1)

    async def test_missing_optional_fields(self):
        doi = "10.1234/example.2024.001"
        sparse = {
            "status": "ok",
            "message": {
                "DOI": doi,
            },
        }
        with respx.mock(assert_all_called=True) as router:
            router.get(f"{CROSSREF_BASE_URL}/{doi}").mock(
                return_value=httpx.Response(200, json=sparse)
            )
            result = await resolve_doi(doi)

        self.assertIsNone(result.title)
        self.assertEqual(result.authors, ())
        self.assertIsNone(result.journal)
        self.assertIsNone(result.publisher)
        self.assertIsNone(result.published_at)
        self.assertIsNone(result.primary_url)

    async def test_malformed_doi_raises(self):
        with self.assertRaises(ValueError):
            await resolve_doi("not-a-doi")

    async def test_404_propagates(self):
        doi = "10.1234/example.2024.999"
        with respx.mock(assert_all_called=True) as router:
            router.get(f"{CROSSREF_BASE_URL}/{doi}").mock(
                return_value=httpx.Response(404, json={"status": "no"})
            )
            with self.assertRaises(httpx.HTTPStatusError):
                await resolve_doi(doi)

    async def test_invalid_date_returns_none(self):
        doi = "10.1234/example.2024.001"
        with respx.mock(assert_all_called=True) as router:
            router.get(f"{CROSSREF_BASE_URL}/{doi}").mock(
                return_value=httpx.Response(
                    200,
                    json=_crossref_payload(
                        issued={"date-parts": [[2024, 13, 99]]}
                    ),
                )
            )
            result = await resolve_doi(doi)
        self.assertIsNone(result.published_at)

    async def test_author_with_only_family(self):
        doi = "10.1234/example.2024.001"
        with respx.mock(assert_all_called=True) as router:
            router.get(f"{CROSSREF_BASE_URL}/{doi}").mock(
                return_value=httpx.Response(
                    200,
                    json=_crossref_payload(author=[{"family": "OnlyFamily"}]),
                )
            )
            result = await resolve_doi(doi)
        self.assertEqual(result.authors, ("OnlyFamily",))
