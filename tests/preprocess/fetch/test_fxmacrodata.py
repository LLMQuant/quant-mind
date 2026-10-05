"""Tests for preprocess.fetch.fxmacrodata."""

import unittest

import httpx
import respx

from quantmind.preprocess.fetch.fxmacrodata import fetch_fxmacrodata_calendar

_URL = "https://api.fxmacrodata.com/v1/calendar/usd"
_KEY = "test-key"


class FetchFxmacrodataCalendarTests(unittest.IsolatedAsyncioTestCase):
    async def test_parses_rows_and_sends_key_header(self):
        with respx.mock(assert_all_called=True) as router:
            route = router.get(_URL).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "currency": "usd",
                        "data": [{"release": "cpi", "name": "CPI"}],
                    },
                )
            )
            calendar = await fetch_fxmacrodata_calendar(api_key=_KEY)

        self.assertEqual(route.calls[0].request.headers["X-API-Key"], _KEY)
        self.assertEqual(calendar.currency, "USD")
        self.assertEqual(calendar.releases[0].release, "cpi")

    async def test_redirect_is_not_followed_with_key(self):
        with respx.mock(assert_all_called=False) as router:
            router.get(_URL).mock(
                return_value=httpx.Response(
                    302, headers={"Location": "https://other.example/x"}
                )
            )
            other = router.get("https://other.example/x").mock(
                return_value=httpx.Response(200, json={"data": []})
            )
            with self.assertRaises(httpx.HTTPStatusError):
                await fetch_fxmacrodata_calendar(api_key=_KEY)

        self.assertFalse(other.called)

    async def test_key_with_whitespace_is_rejected_without_echo(self):
        with self.assertRaises(ValueError) as ctx:
            await fetch_fxmacrodata_calendar(api_key="test key")
        self.assertNotIn("test key", str(ctx.exception))

    async def test_error_body_with_200_raises_clean_error(self):
        for body in ({"detail": "Invalid API key"}, [], {"data": "x"}):
            with respx.mock() as router:
                router.get(_URL).mock(
                    return_value=httpx.Response(200, json=body)
                )
                with self.assertRaises(ValueError):
                    await fetch_fxmacrodata_calendar(api_key=_KEY)
