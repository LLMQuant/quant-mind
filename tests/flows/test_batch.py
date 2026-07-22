"""Tests for ``quantmind.flows.batch``."""

import asyncio
import unittest
from dataclasses import dataclass
from typing import Any

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import RawText
from quantmind.flows._usage import PriceRate, record_usage
from quantmind.flows.batch import (
    BatchResult,
    BudgetExceededError,
    batch_run,
)


@dataclass
class _FakeUsage:
    """Duck-types the SDK ``Usage`` object for tests."""

    requests: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class BatchResultPropertiesTests(unittest.TestCase):
    def test_successes_and_failures_views(self) -> None:
        result: BatchResult[str] = BatchResult(
            total=4,
            success_count=2,
            failure_count=2,
            results=["a", None, "c", None],
            errors=[(1, ValueError("b")), (3, RuntimeError("d"))],
            duration_seconds=0.0,
        )
        self.assertEqual(result.successes, [(0, "a"), (2, "c")])
        self.assertEqual(
            [(i, type(e).__name__) for i, e in result.failures],
            [(1, "ValueError"), (3, "RuntimeError")],
        )


class BatchRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_happy_path(self) -> None:
        async def flow(input: RawText, *, cfg: Any = None) -> str:
            return f"ok:{input.text}"

        inputs = [RawText(text=str(i)) for i in range(5)]
        result = await batch_run(flow, inputs, concurrency=3)
        self.assertEqual(result.total, 5)
        self.assertEqual(result.success_count, 5)
        self.assertEqual(result.failure_count, 0)
        self.assertEqual(result.results, [f"ok:{i}" for i in range(5)])
        self.assertEqual(result.errors, [])
        self.assertGreaterEqual(result.duration_seconds, 0)
        self.assertEqual(result.tokens_total, {})
        self.assertEqual(result.cost_estimate_usd, 0.0)

    async def test_empty_inputs(self) -> None:
        async def flow(input: RawText, *, cfg: Any = None) -> str:
            return "x"

        result = await batch_run(flow, [], concurrency=4)
        self.assertEqual(result.total, 0)
        self.assertEqual(result.success_count, 0)
        self.assertEqual(result.results, [])

    async def test_on_error_skip_collects(self) -> None:
        async def flow(input: RawText, *, cfg: Any = None) -> str:
            if input.text in ("2", "4"):
                raise ValueError(f"bad:{input.text}")
            return f"ok:{input.text}"

        inputs = [RawText(text=str(i)) for i in range(5)]
        result = await batch_run(flow, inputs, on_error="skip")
        self.assertEqual(result.success_count, 3)
        self.assertEqual(result.failure_count, 2)
        self.assertEqual(result.results[0], "ok:0")
        self.assertIsNone(result.results[2])
        self.assertIsNone(result.results[4])
        self.assertEqual([i for i, _ in result.errors], [2, 4])

    async def test_on_error_raise_propagates(self) -> None:
        boom = RuntimeError("boom")

        async def flow(input: RawText, *, cfg: Any = None) -> str:
            raise boom

        with self.assertRaises(RuntimeError) as ctx:
            await batch_run(
                flow,
                [RawText(text="x")],
                concurrency=1,
                on_error="raise",
            )
        self.assertIs(ctx.exception, boom)

    async def test_memory_kwarg_rejected(self) -> None:
        async def flow(input: RawText, *, cfg: Any = None) -> str:
            return "x"

        with self.assertRaises(ValueError) as ctx:
            await batch_run(flow, [RawText(text="x")], memory=object())
        self.assertIn("memory", str(ctx.exception))

    async def test_concurrency_must_be_at_least_one(self) -> None:
        async def flow(input: RawText, *, cfg: Any = None) -> str:
            return "x"

        with self.assertRaises(ValueError):
            await batch_run(flow, [RawText(text="x")], concurrency=0)

    async def test_concurrency_cap_honoured(self) -> None:
        in_flight = 0
        peak = 0

        async def flow(input: RawText, *, cfg: Any = None) -> str:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)  # let scheduler interleave
            in_flight -= 1
            return "ok"

        await batch_run(
            flow,
            [RawText(text=str(i)) for i in range(10)],
            concurrency=3,
        )
        self.assertLessEqual(peak, 3)
        self.assertGreater(peak, 0)

    async def test_on_progress_called_per_completion(self) -> None:
        calls: list[tuple[int, int]] = []

        async def flow(input: RawText, *, cfg: Any = None) -> str:
            return "ok"

        inputs = [RawText(text=str(i)) for i in range(5)]
        await batch_run(
            flow,
            inputs,
            concurrency=2,
            on_progress=lambda done, total: calls.append((done, total)),
        )
        self.assertEqual(len(calls), 5)
        # `done` strictly increases, ends at total.
        dones = [c[0] for c in calls]
        self.assertEqual(dones, sorted(dones))
        self.assertEqual(dones[-1], 5)
        self.assertTrue(all(t == 5 for _, t in calls))

    async def test_cfg_forwarded(self) -> None:
        seen_cfg: list[Any] = []

        async def flow(input: RawText, *, cfg: Any = None) -> str:
            seen_cfg.append(cfg)
            return "ok"

        cfg = PaperFlowCfg(model="sentinel-model")
        await batch_run(flow, [RawText(text="x")], cfg=cfg, concurrency=1)
        self.assertIs(seen_cfg[0], cfg)

    async def test_extra_kwargs_forwarded(self) -> None:
        seen: list[Any] = []

        async def flow(
            input: RawText, *, cfg: Any = None, marker: str = ""
        ) -> str:
            seen.append(marker)
            return "ok"

        await batch_run(flow, [RawText(text="x")], concurrency=1, marker="here")
        self.assertEqual(seen, ["here"])


class BatchUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_usage_aggregated_across_inputs(self) -> None:
        async def flow(input: RawText, *, cfg: Any = None) -> str:
            record_usage(
                _FakeUsage(input_tokens=10, output_tokens=4, total_tokens=14)
            )
            return f"ok:{input.text}"

        inputs = [RawText(text=str(i)) for i in range(3)]
        result = await batch_run(flow, inputs, concurrency=2)
        self.assertEqual(
            result.tokens_total,
            {
                "requests": 3,
                "input_tokens": 30,
                "output_tokens": 12,
                "total_tokens": 42,
            },
        )
        self.assertEqual(result.cost_estimate_usd, 0.0)

    async def test_prices_populate_cost_estimate(self) -> None:
        async def flow(input: RawText, *, cfg: Any = None) -> str:
            record_usage(_FakeUsage(input_tokens=1_000_000, output_tokens=0))
            return "ok"

        cfg = PaperFlowCfg(model="gpt-test")
        prices = {
            "gpt-test": PriceRate(input_usd_per_1m=2.0, output_usd_per_1m=8.0)
        }
        result = await batch_run(
            flow, [RawText(text="x")], cfg=cfg, concurrency=1, prices=prices
        )
        # 1M input tokens @ $2/1M = $2.00.
        self.assertAlmostEqual(result.cost_estimate_usd, 2.0)

    async def test_token_budget_trips_and_skips_later_inputs(self) -> None:
        async def flow(input: RawText, *, cfg: Any = None) -> str:
            record_usage(_FakeUsage(input_tokens=100))
            return f"ok:{input.text}"

        cfg = PaperFlowCfg(model="gpt-test", max_total_input_tokens=150)
        inputs = [RawText(text=str(i)) for i in range(4)]
        # concurrency=1 makes the trip point deterministic: after input 1 the
        # running total (200) exceeds 150, so inputs 2 and 3 are skipped.
        result = await batch_run(flow, inputs, cfg=cfg, concurrency=1)
        self.assertEqual(result.success_count, 2)
        self.assertEqual([i for i, _ in result.errors], [2, 3])
        self.assertTrue(
            all(isinstance(e, BudgetExceededError) for _, e in result.errors)
        )

    async def test_cost_budget_trips_with_prices(self) -> None:
        async def flow(input: RawText, *, cfg: Any = None) -> str:
            record_usage(_FakeUsage(input_tokens=1_000_000))
            return "ok"

        cfg = PaperFlowCfg(model="gpt-test", max_total_cost_usd=1.5)
        prices = {
            "gpt-test": PriceRate(input_usd_per_1m=1.0, output_usd_per_1m=1.0)
        }
        inputs = [RawText(text=str(i)) for i in range(4)]
        result = await batch_run(
            flow, inputs, cfg=cfg, concurrency=1, prices=prices
        )
        # $1/input; after input 1 cumulative $2.00 > $1.50 → skip 2 and 3.
        self.assertEqual(result.success_count, 2)
        self.assertEqual([i for i, _ in result.errors], [2, 3])

    async def test_budget_raise_propagates(self) -> None:
        async def flow(input: RawText, *, cfg: Any = None) -> str:
            record_usage(_FakeUsage(input_tokens=100))
            return "ok"

        cfg = PaperFlowCfg(model="gpt-test", max_total_input_tokens=50)
        inputs = [RawText(text=str(i)) for i in range(3)]
        with self.assertRaises(BudgetExceededError):
            await batch_run(
                flow, inputs, cfg=cfg, concurrency=1, on_error="raise"
            )
