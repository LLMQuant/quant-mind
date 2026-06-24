"""Tests for ``quantmind.flows.token_dashboard``."""

import unittest

from agents.items import ModelResponse
from agents.usage import Usage

from quantmind.flows import SessionTokenDashboard


def _make_response(
    *,
    requests: int,
    input_tokens: int,
    output_tokens: int,
) -> ModelResponse:
    """Build a minimal model response with usage for dashboard tests."""
    return ModelResponse(
        output=[],
        usage=Usage(
            requests=requests,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        response_id=None,
        request_id=None,
    )


class SessionTokenDashboardTests(unittest.IsolatedAsyncioTestCase):
    async def test_on_llm_end_accumulates_usage(self) -> None:
        dashboard = SessionTokenDashboard(session_token_budget=500)
        await dashboard.on_llm_end(
            context=None,
            agent=None,
            response=_make_response(
                requests=1,
                input_tokens=100,
                output_tokens=25,
            ),
        )
        await dashboard.on_llm_end(
            context=None,
            agent=None,
            response=_make_response(
                requests=1,
                input_tokens=60,
                output_tokens=15,
            ),
        )
        snapshot = dashboard.snapshot()
        self.assertEqual(snapshot.requests, 2)
        self.assertEqual(snapshot.input_tokens, 160)
        self.assertEqual(snapshot.output_tokens, 40)
        self.assertEqual(snapshot.total_tokens, 200)
        self.assertEqual(snapshot.remaining_tokens, 300)
        self.assertEqual(snapshot.used_percent, 40.0)

    async def test_reset_clears_all_usage_counters(self) -> None:
        dashboard = SessionTokenDashboard(session_token_budget=100)
        await dashboard.on_llm_end(
            context=None,
            agent=None,
            response=_make_response(
                requests=1,
                input_tokens=80,
                output_tokens=10,
            ),
        )
        dashboard.reset()
        snapshot = dashboard.snapshot()
        self.assertEqual(snapshot.requests, 0)
        self.assertEqual(snapshot.input_tokens, 0)
        self.assertEqual(snapshot.output_tokens, 0)
        self.assertEqual(snapshot.total_tokens, 0)
        self.assertEqual(snapshot.remaining_tokens, 100)
        self.assertEqual(snapshot.used_percent, 0.0)

    def test_snapshot_without_budget_has_no_remaining(self) -> None:
        dashboard = SessionTokenDashboard()
        snapshot = dashboard.snapshot()
        self.assertIsNone(snapshot.session_token_budget)
        self.assertIsNone(snapshot.remaining_tokens)
        self.assertIsNone(snapshot.used_percent)

    def test_render_with_budget_includes_progress_and_remaining(self) -> None:
        dashboard = SessionTokenDashboard(session_token_budget=200)
        dashboard._usage.add(  # pyright: ignore[reportPrivateUsage]
            Usage(
                requests=2,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            )
        )
        rendered = dashboard.render(bar_width=10)
        self.assertIn("Session Token Dashboard", rendered)
        self.assertIn("Remaining: 50", rendered)
        self.assertIn("Used/Budget: 150/200", rendered)
        self.assertIn("75.0% used", rendered)

    def test_render_without_budget_mentions_missing_budget(self) -> None:
        dashboard = SessionTokenDashboard()
        rendered = dashboard.render()
        self.assertIn("Remaining: n/a (set session_token_budget)", rendered)

    def test_as_dict_returns_serializable_snapshot(self) -> None:
        dashboard = SessionTokenDashboard(session_token_budget=300)
        dashboard._usage.add(  # pyright: ignore[reportPrivateUsage]
            Usage(
                requests=1,
                input_tokens=90,
                output_tokens=30,
                total_tokens=120,
            )
        )
        payload = dashboard.as_dict()
        self.assertEqual(
            payload,
            {
                "requests": 1,
                "input_tokens": 90,
                "output_tokens": 30,
                "total_tokens": 120,
                "session_token_budget": 300,
                "remaining_tokens": 180,
                "used_percent": 40.0,
            },
        )


class SessionTokenDashboardValidationTests(unittest.TestCase):
    def test_invalid_budget_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            SessionTokenDashboard(session_token_budget=0)

    def test_invalid_render_width_raises_value_error(self) -> None:
        dashboard = SessionTokenDashboard(session_token_budget=100)
        with self.assertRaises(ValueError):
            dashboard.render(bar_width=0)

    def test_set_session_token_budget_validates_values(self) -> None:
        dashboard = SessionTokenDashboard(session_token_budget=100)
        dashboard.set_session_token_budget(250)
        self.assertEqual(dashboard.session_token_budget, 250)
        with self.assertRaises(ValueError):
            dashboard.set_session_token_budget(-10)
