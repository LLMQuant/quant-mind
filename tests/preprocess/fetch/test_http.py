"""Tests for preprocess.fetch.http — fetch_url."""

import unittest

import httpx
import respx

from quantmind.preprocess.fetch.http import fetch_url


class FetchUrlTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_body_and_metadata(self):
        with respx.mock(assert_all_called=True) as router:
            router.get("https://example.com/data").mock(
                return_value=httpx.Response(
                    200,
                    headers={
                        "Content-Type": "text/plain; charset=utf-8",
                        "ETag": "abc123",
                        "X-Ignored": "ignored",
                    },
                    content=b"hello world",
                )
            )
            result = await fetch_url("https://example.com/data")

        self.assertEqual(result.bytes, b"hello world")
        self.assertEqual(result.content_type, "text/plain")
        self.assertEqual(result.source_url, "https://example.com/data")
        self.assertEqual(result.headers.get("etag"), "abc123")
        self.assertNotIn("x-ignored", result.headers)

    async def test_default_user_agent_sent(self):
        with respx.mock(assert_all_called=True) as router:
            route = router.get("https://example.com").mock(
                return_value=httpx.Response(200, content=b"")
            )
            await fetch_url("https://example.com")

        sent_request = route.calls.last.request
        self.assertIn("QuantMind", sent_request.headers["User-Agent"])

    async def test_max_bytes_overflow_raises(self):
        with respx.mock(assert_all_called=True) as router:
            router.get("https://example.com").mock(
                return_value=httpx.Response(200, content=b"x" * 50)
            )
            with self.assertRaises(ValueError):
                await fetch_url("https://example.com", max_bytes=10)

    async def test_http_error_propagates(self):
        with respx.mock(assert_all_called=True) as router:
            router.get("https://example.com").mock(
                return_value=httpx.Response(404)
            )
            with self.assertRaises(httpx.HTTPStatusError):
                await fetch_url("https://example.com")

    async def test_missing_content_type_defaults_to_octet_stream(self):
        with respx.mock(assert_all_called=True) as router:
            router.get("https://example.com").mock(
                return_value=httpx.Response(200, content=b"")
            )
            result = await fetch_url("https://example.com")

        self.assertEqual(result.content_type, "application/octet-stream")
