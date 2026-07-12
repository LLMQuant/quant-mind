"""Tests for preprocess.fetch.http."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import respx

from quantmind.preprocess.fetch.http import (
    FetchAttemptsExhausted,
    FetchPolicy,
    HttpFetcher,
    fetch_url,
)


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
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.resolved_url, "https://example.com/data")
        self.assertIsNotNone(result.fetched_at)
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


class FetchPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_transient_status_with_backoff(self):
        policy = FetchPolicy(
            max_attempts=3,
            backoff_base_seconds=0.25,
            backoff_max_seconds=1.0,
            jitter_seconds=0.0,
        )
        with (
            respx.mock(assert_all_called=True) as router,
            patch(
                "quantmind.preprocess.fetch.http.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
        ):
            route = router.get("https://example.com/transient").mock(
                side_effect=[
                    httpx.Response(503),
                    httpx.Response(200, content=b"ok"),
                ]
            )
            result = await fetch_url(
                "https://example.com/transient",
                fetch_policy=policy,
            )

        self.assertEqual(result.bytes, b"ok")
        self.assertEqual(route.call_count, 2)
        sleep.assert_awaited_once_with(0.25)

    async def test_retry_after_overrides_backoff(self):
        policy = FetchPolicy(
            max_attempts=2,
            backoff_base_seconds=0.1,
            backoff_max_seconds=1.0,
            jitter_seconds=0.0,
        )
        with (
            respx.mock(assert_all_called=True) as router,
            patch(
                "quantmind.preprocess.fetch.http.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
        ):
            router.get("https://example.com/rate-limited").mock(
                side_effect=[
                    httpx.Response(429, headers={"Retry-After": "2"}),
                    httpx.Response(200, content=b"ok"),
                ]
            )
            await fetch_url(
                "https://example.com/rate-limited",
                fetch_policy=policy,
            )

        sleep.assert_awaited_once_with(2.0)

    async def test_retries_transport_failure(self):
        policy = FetchPolicy(
            max_attempts=2,
            backoff_base_seconds=0.0,
            backoff_max_seconds=0.0,
            jitter_seconds=0.0,
        )
        with (
            respx.mock(assert_all_called=True) as router,
            patch(
                "quantmind.preprocess.fetch.http.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            route = router.get("https://example.com/network").mock(
                side_effect=[
                    httpx.ConnectError("connection reset"),
                    httpx.Response(200, content=b"recovered"),
                ]
            )
            result = await fetch_url(
                "https://example.com/network",
                fetch_policy=policy,
            )

        self.assertEqual(route.call_count, 2)
        self.assertEqual(result.bytes, b"recovered")

    async def test_retry_exhaustion_has_stable_error(self):
        policy = FetchPolicy(
            max_attempts=3,
            backoff_base_seconds=0.0,
            backoff_max_seconds=0.0,
            jitter_seconds=0.0,
        )
        with (
            respx.mock(assert_all_called=True) as router,
            patch(
                "quantmind.preprocess.fetch.http.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            route = router.get("https://example.com/down").mock(
                return_value=httpx.Response(500)
            )
            with self.assertRaises(FetchAttemptsExhausted) as context:
                await fetch_url(
                    "https://example.com/down",
                    fetch_policy=policy,
                )

        self.assertEqual(route.call_count, 3)
        self.assertEqual(context.exception.attempts, 3)
        self.assertIsInstance(
            context.exception.last_error,
            httpx.HTTPStatusError,
        )

    async def test_non_retryable_status_is_one_shot(self):
        with respx.mock(assert_all_called=True) as router:
            route = router.get("https://example.com/not-found").mock(
                return_value=httpx.Response(404)
            )
            with self.assertRaises(httpx.HTTPStatusError):
                await fetch_url(
                    "https://example.com/not-found",
                    fetch_policy=FetchPolicy(),
                )

        self.assertEqual(route.call_count, 1)

    async def test_shared_fetcher_spaces_same_host_requests(self):
        policy = FetchPolicy(
            max_attempts=1,
            jitter_seconds=0.0,
            min_interval_seconds=0.5,
        )
        with (
            respx.mock(assert_all_called=True) as router,
            patch(
                "quantmind.preprocess.fetch.http.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
        ):
            router.get(url__regex=r"https://example\.com/.*").mock(
                return_value=httpx.Response(200)
            )
            async with HttpFetcher(policy=policy) as fetcher:
                await fetcher.fetch_url("https://example.com/one")
                await fetcher.fetch_url("https://example.com/two")

        self.assertEqual(sleep.await_count, 1)
        self.assertGreater(sleep.await_args.args[0], 0.4)

    async def test_shared_fetcher_caps_same_host_concurrency(self):
        in_flight = 0
        peak = 0

        async def respond(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return httpx.Response(200, request=request)

        policy = FetchPolicy(
            max_attempts=1,
            max_concurrency_per_host=1,
        )
        with respx.mock(assert_all_called=True) as router:
            router.get(url__regex=r"https://example\.com/.*").mock(
                side_effect=respond
            )
            async with HttpFetcher(policy=policy) as fetcher:
                await asyncio.gather(
                    fetcher.fetch_url("https://example.com/one"),
                    fetcher.fetch_url("https://example.com/two"),
                )

        self.assertEqual(peak, 1)


class FetchPolicyValidationTests(unittest.TestCase):
    def test_rejects_invalid_attempt_count(self):
        with self.assertRaises(ValueError):
            FetchPolicy(max_attempts=0)
