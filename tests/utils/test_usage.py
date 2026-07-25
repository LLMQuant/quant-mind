"""Tests for per-run usage collection from Agents SDK spans.

Spans are hand-built so the suite runs fully offline: no model call, no real
trace export. ``usage_scope`` is driven with a patched ``trace`` and spans
pushed straight into the collector, simulating what the SDK would record.
"""

import unittest
from typing import Any
from unittest.mock import patch

from quantmind.utils import usage
from quantmind.utils.usage import RunUsage, SpanCollector, usage_scope


class _SpanData:
    """Stand-in for an SDK ``*SpanData`` — exposes ``.type`` plus given attrs."""

    def __init__(self, type_: str, **attrs: Any) -> None:
        self._type = type_
        for key, value in attrs.items():
            setattr(self, key, value)

    @property
    def type(self) -> str:
        return self._type


class _Span:
    """Stand-in for an SDK ``Span``."""

    def __init__(
        self,
        *,
        span_id: str,
        parent_id: str | None,
        trace_id: str,
        started_at: str | None,
        ended_at: str | None,
        span_data: _SpanData,
    ) -> None:
        self.span_id = span_id
        self.parent_id = parent_id
        self.trace_id = trace_id
        self.started_at = started_at
        self.ended_at = ended_at
        self.span_data = span_data


class _RespUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = None  # force the input+output fallback


class _Response:
    def __init__(self, model: str, usage: _RespUsage) -> None:
        self.model = model
        self.usage = usage


def _agent(span_id: str, name: str, trace_id: str = "t1") -> _Span:
    return _Span(
        span_id=span_id,
        parent_id=None,
        trace_id=trace_id,
        started_at="2026-07-22T10:00:00+00:00",
        ended_at="2026-07-22T10:00:05+00:00",
        span_data=_SpanData("agent", name=name),
    )


def _generation(
    span_id: str,
    parent_id: str | None,
    *,
    start: str,
    end: str,
    usage_dict: dict[str, Any] | None,
    model: str | None = "gpt-4o",
    trace_id: str = "t1",
) -> _Span:
    return _Span(
        span_id=span_id,
        parent_id=parent_id,
        trace_id=trace_id,
        started_at=start,
        ended_at=end,
        span_data=_SpanData("generation", usage=usage_dict, model=model),
    )


class AggregateTests(unittest.TestCase):
    """The core reduce: leaf model spans only, labels, and token sums."""

    def test_sums_generation_and_response_and_skips_aggregate_spans(
        self,
    ) -> None:
        agent = _agent("a1", "paper_extractor")
        gen = _generation(
            "g1",
            "a1",
            start="2026-07-22T10:00:00+00:00",
            end="2026-07-22T10:00:02+00:00",
            usage_dict={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        resp = _Span(
            span_id="r1",
            parent_id="a1",
            trace_id="t1",
            started_at="2026-07-22T10:00:02+00:00",
            ended_at="2026-07-22T10:00:04+00:00",
            # response span: usage on response.usage, no top-level dict, no model
            span_data=_SpanData(
                "response",
                usage=None,
                response=_Response("gpt-4o-mini", _RespUsage(200, 80)),
            ),
        )
        # A task-level aggregate span that must NOT be summed (double-count trap).
        task = _Span(
            span_id="k1",
            parent_id=None,
            trace_id="t1",
            started_at="2026-07-22T10:00:00+00:00",
            ended_at="2026-07-22T10:00:05+00:00",
            span_data=_SpanData("task", usage={"input_tokens": 9999}),
        )

        result = usage._aggregate([task, agent, gen, resp])

        self.assertEqual(result.requests, 2)  # only the two model spans
        self.assertEqual(result.input_tokens, 300)
        self.assertEqual(result.output_tokens, 130)
        self.assertEqual(result.total_tokens, 430)  # 150 + (200 + 80)
        self.assertEqual(
            [s.label for s in result.steps], ["paper_extractor"] * 2
        )
        self.assertEqual(
            [s.model for s in result.steps], ["gpt-4o", "gpt-4o-mini"]
        )
        self.assertEqual(result.steps[0].span_id, "g1")  # sorted by start time

    def test_wall_is_span_wide_while_busy_is_per_step_sum(self) -> None:
        # Two overlapping calls: 10:00:00-04 (4s) and 10:00:01-03 (2s).
        a = _generation(
            "g1",
            None,
            start="2026-07-22T10:00:00+00:00",
            end="2026-07-22T10:00:04+00:00",
            usage_dict={"input_tokens": 1, "output_tokens": 1},
        )
        b = _generation(
            "g2",
            None,
            start="2026-07-22T10:00:01+00:00",
            end="2026-07-22T10:00:03+00:00",
            usage_dict={"input_tokens": 1, "output_tokens": 1},
        )

        result = usage._aggregate([a, b])

        self.assertEqual(result.busy_seconds, 6.0)  # 4 + 2, sum of durations
        self.assertEqual(result.wall_seconds, 4.0)  # max end - min start

    def test_label_falls_back_to_span_type_without_agent_ancestor(self) -> None:
        gen = _generation(
            "g1",
            None,
            start="2026-07-22T10:00:00+00:00",
            end="2026-07-22T10:00:01+00:00",
            usage_dict={"input_tokens": 5, "output_tokens": 5},
        )
        result = usage._aggregate([gen])
        self.assertEqual(result.steps[0].label, "generation")

    def test_empty_trace_yields_zeroed_usage(self) -> None:
        result = usage._aggregate([])
        self.assertEqual(result, RunUsage.empty())


class UsageExtractionTests(unittest.TestCase):
    """Provider-shape tolerance and timestamp parsing."""

    def test_prompt_completion_keys_map_to_input_output(self) -> None:
        span_data = _SpanData(
            "generation",
            usage={"prompt_tokens": 12, "completion_tokens": 8},
            model="deepseek-chat",
        )
        self.assertEqual(usage._extract_usage(span_data), (12, 8, 20))

    def test_missing_usage_returns_none(self) -> None:
        span_data = _SpanData("generation", usage=None, model="x")
        self.assertIsNone(usage._extract_usage(span_data))

    def test_parse_iso_accepts_z_suffix_and_rejects_garbage(self) -> None:
        self.assertIsNotNone(usage._parse_iso("2026-07-22T10:00:00Z"))
        self.assertIsNone(usage._parse_iso("not-a-timestamp"))
        self.assertIsNone(usage._parse_iso(None))


class SpanCollectorTests(unittest.TestCase):
    """Bucketing and eviction by trace id."""

    def test_buckets_by_trace_id_and_pop_clears(self) -> None:
        collector = SpanCollector()
        collector.on_span_end(_agent("a1", "x", trace_id="t1"))
        collector.on_span_end(_agent("a2", "y", trace_id="t1"))
        collector.on_span_end(_agent("a3", "z", trace_id="t2"))

        first = collector.pop("t1")
        self.assertEqual(len(first), 2)
        self.assertEqual(collector.pop("t1"), [])  # evicted
        self.assertEqual(len(collector.pop("t2")), 1)

    def test_no_op_lifecycle_methods_do_not_raise(self) -> None:
        collector = SpanCollector()
        self.assertIsNone(collector.on_trace_start(object()))
        self.assertIsNone(collector.on_trace_end(object()))
        self.assertIsNone(collector.on_span_start(object()))
        self.assertIsNone(collector.shutdown())
        self.assertIsNone(collector.force_flush())


class GetCollectorTests(unittest.TestCase):
    """The global collector is a singleton, registered exactly once."""

    def setUp(self) -> None:
        usage._collector = None

    def tearDown(self) -> None:
        usage._collector = None

    def test_singleton_registered_once(self) -> None:
        with patch.object(usage, "add_trace_processor") as add:
            first = usage._get_collector()
            second = usage._get_collector()
        self.assertIs(first, second)
        add.assert_called_once_with(first)


class UsageScopeTests(unittest.TestCase):
    """End-to-end scope behavior with a patched trace and simulated spans."""

    def setUp(self) -> None:
        usage._collector = None
        self._add = patch.object(usage, "add_trace_processor").start()
        self._trace = patch.object(usage, "trace").start()

    def tearDown(self) -> None:
        patch.stopall()
        usage._collector = None

    def test_populates_usage_on_exit_and_evicts(self) -> None:
        with usage_scope("quantmind.paper") as run:
            trace_id = run.trace_id
            collector = usage._get_collector()
            collector.on_span_end(
                _generation(
                    "g1",
                    None,
                    start="2026-07-22T10:00:00+00:00",
                    end="2026-07-22T10:00:01+00:00",
                    usage_dict={"input_tokens": 10, "output_tokens": 4},
                    trace_id=trace_id,
                )
            )
            self.assertEqual(run.usage, RunUsage.empty())  # not yet aggregated

        self.assertEqual(run.usage.requests, 1)
        self.assertEqual(run.usage.total_tokens, 14)
        self.assertEqual(usage._get_collector().pop(trace_id), [])  # evicted
        self._trace.assert_called_once()

    def test_distinct_trace_id_per_scope(self) -> None:
        with usage_scope() as a, usage_scope() as b:
            self.assertNotEqual(a.trace_id, b.trace_id)

    def test_usage_populated_even_when_block_raises(self) -> None:
        scope_ref = {}
        with self.assertRaises(ValueError):
            with usage_scope() as run:
                scope_ref["run"] = run
                usage._get_collector().on_span_end(
                    _generation(
                        "g1",
                        None,
                        start="2026-07-22T10:00:00+00:00",
                        end="2026-07-22T10:00:01+00:00",
                        usage_dict={"input_tokens": 7, "output_tokens": 3},
                        trace_id=run.trace_id,
                    )
                )
                raise ValueError("boom")

        self.assertEqual(scope_ref["run"].usage.total_tokens, 10)


if __name__ == "__main__":
    unittest.main()
