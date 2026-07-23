"""Tests for ``quantmind.flows._usage``."""

import asyncio
import unittest
from dataclasses import dataclass

from quantmind.flows._usage import (
    PriceRate,
    UsageSummary,
    record_usage,
    usage_scope,
)


@dataclass
class _FakeUsage:
    """Duck-types the SDK ``Usage`` object for tests."""

    requests: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class UsageSummaryTests(unittest.TestCase):
    def test_as_tokens_dict_excludes_cost(self) -> None:
        summary = UsageSummary(
            requests=2, input_tokens=10, output_tokens=3, total_tokens=13
        )
        self.assertEqual(
            summary.as_tokens_dict(),
            {
                "requests": 2,
                "input_tokens": 10,
                "output_tokens": 3,
                "total_tokens": 13,
            },
        )


class PriceRateTests(unittest.TestCase):
    def test_cost_is_per_million(self) -> None:
        rate = PriceRate(input_usd_per_1m=300.0, output_usd_per_1m=600.0)
        # 1M input @ 300 + 0.5M output @ 600 = 300 + 300 = 600.
        self.assertAlmostEqual(rate.cost(1_000_000, 500_000), 600.0)


class UsageScopeTests(unittest.IsolatedAsyncioTestCase):
    def test_record_outside_scope_is_noop(self) -> None:
        # Must not raise when no scope is active.
        record_usage(_FakeUsage(input_tokens=99))

    def test_accumulator_sums_multiple_records(self) -> None:
        with usage_scope() as acc:
            record_usage(_FakeUsage(input_tokens=5, output_tokens=2))
            record_usage(_FakeUsage(input_tokens=3, output_tokens=1))
        summary = acc.summary()
        self.assertEqual(summary.requests, 2)
        self.assertEqual(summary.input_tokens, 8)
        self.assertEqual(summary.output_tokens, 3)

    async def test_scopes_isolate_across_concurrent_tasks(self) -> None:
        async def worker(unit: int) -> int:
            with usage_scope() as acc:
                record_usage(_FakeUsage(input_tokens=unit))
                await asyncio.sleep(0)  # force interleave with the sibling
                record_usage(_FakeUsage(input_tokens=unit))
                return acc.summary().input_tokens

        one, ten = await asyncio.gather(worker(1), worker(10))
        self.assertEqual(one, 2)
        self.assertEqual(ten, 20)

    async def test_nested_gather_accumulates_into_outer_scope(self) -> None:
        # The batch case: children run as separate tasks yet fold into the
        # one accumulator the outer scope set (shared mutable reference).
        async def child(unit: int) -> None:
            record_usage(_FakeUsage(input_tokens=unit))

        with usage_scope() as acc:
            await asyncio.gather(child(3), child(4))
        self.assertEqual(acc.summary().input_tokens, 7)


if __name__ == "__main__":
    unittest.main()
