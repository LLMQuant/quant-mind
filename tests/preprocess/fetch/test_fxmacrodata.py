"""Tests for preprocess.fetch.fxmacrodata."""

import unittest
import traceback

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

    async def test_error_envelope_with_data_is_rejected(self):
        with respx.mock() as router:
            router.get(_URL).mock(
                return_value=httpx.Response(
                    200, json={"error": "failed", "data": []}
                )
            )
            with self.assertRaises(ValueError):
                await fetch_fxmacrodata_calendar(api_key=_KEY)

    async def test_malformed_rows_are_not_silently_dropped(self):
        with respx.mock() as router:
            router.get(_URL).mock(
                return_value=httpx.Response(
                    200, json={"data": [{"release": "cpi"}, "invalid"]}
                )
            )
            with self.assertRaises(ValueError):
                await fetch_fxmacrodata_calendar(api_key=_KEY)

    async def test_confirmation_requires_boolean_true(self):
        with respx.mock() as router:
            router.get(_URL).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "data": [
                            {"release_date_confirmed": value}
                            for value in ["false", "true", 1, False, True]
                        ]
                    },
                )
            )
            calendar = await fetch_fxmacrodata_calendar(api_key=_KEY)
        self.assertEqual(
            [row.release_date_confirmed for row in calendar.releases],
            [False, False, False, False, True],
        )

    async def test_response_details_are_not_in_error(self):
        with respx.mock() as router:
            router.get(_URL).mock(
                return_value=httpx.Response(
                    200, json={"detail": "sensitive upstream text"}
                )
            )
            with self.assertRaises(ValueError) as caught:
                await fetch_fxmacrodata_calendar(api_key=_KEY)
        self.assertNotIn("sensitive upstream text", str(caught.exception))

    async def test_transport_traceback_is_sanitized(self):
        with respx.mock() as router:
            router.get(_URL).mock(
                side_effect=httpx.ConnectError("sensitive upstream text")
            )
            try:
                await fetch_fxmacrodata_calendar(api_key=_KEY)
            except httpx.RequestError as error:
                self.assertNotIn(
                    "sensitive upstream text",
                    "".join(traceback.format_exception(error)),
                )
            else:
                self.fail("Expected transport failure")

    async def test_control_character_key_rejected_before_request(self):
        with self.assertRaises(ValueError):
            await fetch_fxmacrodata_calendar(api_key="test\x00key")

    async def test_non_json_response_is_sanitized(self):
        with respx.mock() as router:
            router.get(_URL).mock(
                return_value=httpx.Response(200, content=b"invalid")
            )
            with self.assertRaises(ValueError) as caught:
                await fetch_fxmacrodata_calendar(api_key=_KEY)
        self.assertTrue(caught.exception.__suppress_context__)
