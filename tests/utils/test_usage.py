"""Tests for per-run usage collection from Agents SDK spans.

Hand-built spans exercise aggregation. Real SDK traces exercise collection
with an isolated provider: no model call, network request, or trace export.
"""

import asyncio
import unittest
from contextvars import Context
from typing import Any
from unittest.mock import Mock, patch

from agents.tracing import (
    TracingProcessor,
    generation_span,
    get_current_trace,
    trace,
)
from agents.tracing import create as tracing_create
from agents.tracing.provider import DefaultTraceProvider

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
        collector.start("t1")
        collector.start("t2")
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


def _sdk_generation(input_tokens: int = 10, output_tokens: int = 5) -> Any:
    return generation_span(
        model="synthetic-model",
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
    )


class RealSDKUsageScopeTests(unittest.TestCase):
    """Real callbacks without changing the SDK's global provider/exporters."""

    def setUp(self) -> None:
        # DefaultTraceProvider starts empty; only this test's processors are
        # registered. Its import and the create-module patch are test-only SDK
        # internals; public trace/span context managers still run unchanged.
        provider = DefaultTraceProvider()
        provider.set_disabled(False)
        self.observer = Mock(spec=TracingProcessor)
        provider.register_processor(self.observer)
        for patcher in (
            patch.object(
                tracing_create, "get_trace_provider", return_value=provider
            ),
            patch.object(
                usage,
                "add_trace_processor",
                side_effect=provider.register_processor,
            ),
            patch.object(usage, "_collector", None),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(provider.shutdown)
        self.collector = usage._get_collector()
        self.context = Context()  # Do not inherit an ambient trace or span.

    def test_unrelated_traces_are_not_retained_and_observers_still_run(
        self,
    ) -> None:
        def scenario() -> None:
            with usage_scope() as run:
                with _sdk_generation():
                    pass
            for _ in range(3):
                with trace("unrelated"), _sdk_generation(99, 1):
                    pass

            self.assertEqual(run.usage.total_tokens, 15)
            self.assertEqual(run.usage.requests, 1)
            self.assertEqual(self.observer.on_span_end.call_count, 4)
            self.assertEqual(self.collector._by_trace, {})

        self.context.run(scenario)

    def test_late_span_does_not_recreate_a_closed_scope_bucket(self) -> None:
        def scenario() -> None:
            with usage_scope() as run:
                with _sdk_generation():
                    pass
                late_span = _sdk_generation(99, 1)
                late_span.start()

            snapshot = run.usage
            late_span.finish()  # SDK permits span_end after trace_end.
            self.assertEqual(snapshot.total_tokens, 15)
            self.assertEqual(snapshot.requests, 1)
            self.assertIs(run.usage, snapshot)
            self.assertEqual(self.observer.on_span_end.call_count, 2)
            self.assertEqual(self.collector._by_trace, {})

        self.context.run(scenario)

    def test_nested_scopes_restore_the_existing_sdk_trace(self) -> None:
        def scenario() -> None:
            with trace("external") as external, _sdk_generation(99, 1):
                with usage_scope() as outer:
                    with _sdk_generation():
                        pass
                    with usage_scope() as inner:
                        with _sdk_generation(2, 1):
                            pass
                    self.assertEqual(
                        get_current_trace().trace_id, outer.trace_id
                    )
                    with _sdk_generation():
                        pass
                self.assertIs(get_current_trace(), external)

            self.assertNotEqual(outer.trace_id, inner.trace_id)
            self.assertEqual(outer.usage.total_tokens, 30)
            self.assertEqual(outer.usage.requests, 2)
            self.assertEqual(inner.usage.total_tokens, 3)
            self.assertEqual(inner.usage.requests, 1)
            self.assertIsNone(get_current_trace())
            self.assertEqual(self.collector._by_trace, {})

        self.context.run(scenario)

    def test_concurrent_scopes_keep_their_own_usage(self) -> None:
        async def scenario() -> None:
            ready = asyncio.Event()
            entered = 0

            async def worker(tokens: int) -> Any:
                nonlocal entered
                with usage_scope() as run:
                    with _sdk_generation(tokens, 1):
                        entered += 1
                        if entered == 2:
                            ready.set()
                        await ready.wait()
                        self.assertEqual(
                            get_current_trace().trace_id, run.trace_id
                        )
                return run

            first, second = await asyncio.gather(worker(10), worker(20))
            self.assertNotEqual(first.trace_id, second.trace_id)
            self.assertEqual(first.usage.total_tokens, 11)
            self.assertEqual(second.usage.total_tokens, 21)
            self.assertEqual(first.usage.requests, 1)
            self.assertEqual(second.usage.requests, 1)
            self.assertEqual(self.collector._by_trace, {})

        self.context.run(asyncio.run, scenario())

    def test_child_tasks_count_only_when_finished_before_scope_exit(
        self,
    ) -> None:
        async def scenario() -> None:
            started = asyncio.Event()
            release = asyncio.Event()

            async def child() -> None:
                with _sdk_generation(3, 2):
                    started.set()
                    await release.wait()

            with usage_scope() as joined:
                release.set()
                await asyncio.create_task(child())
            self.assertEqual(joined.usage.total_tokens, 5)
            self.assertEqual(joined.usage.requests, 1)

            started.clear()
            release.clear()
            with usage_scope() as exited:
                with _sdk_generation():
                    pass
                task = asyncio.create_task(child())
                await started.wait()
            snapshot = exited.usage
            release.set()
            await task
            self.assertEqual(snapshot.total_tokens, 15)
            self.assertEqual(snapshot.requests, 1)
            self.assertIs(exited.usage, snapshot)
            self.assertEqual(self.collector._by_trace, {})

        self.context.run(asyncio.run, scenario())

    def test_block_and_trace_entry_exceptions_release_the_scope(self) -> None:
        def scenario() -> None:
            with self.assertRaisesRegex(ValueError, "block failed"):
                with usage_scope() as run, _sdk_generation():
                    raise ValueError("block failed")
            self.assertEqual(run.usage.total_tokens, 15)
            self.assertEqual(run.usage.requests, 1)
            self.assertEqual(self.collector._by_trace, {})

            with patch.object(usage, "trace") as broken_trace:
                broken_trace.return_value.__enter__.side_effect = RuntimeError(
                    "trace entry failed"
                )
                with self.assertRaisesRegex(RuntimeError, "trace entry failed"):
                    with usage_scope():
                        self.fail("trace entry should have raised")
            self.assertEqual(self.collector._by_trace, {})

        self.context.run(scenario)

    def test_cancellation_propagates_and_releases_the_scope(self) -> None:
        async def scenario() -> None:
            started = asyncio.Event()
            scopes = []

            async def worker() -> None:
                with usage_scope() as run, _sdk_generation():
                    scopes.append(run)
                    started.set()
                    await asyncio.Event().wait()

            task = asyncio.create_task(worker())
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(scopes[0].usage.total_tokens, 15)
            self.assertEqual(scopes[0].usage.requests, 1)
            self.assertEqual(self.collector._by_trace, {})

        self.context.run(asyncio.run, scenario())


if __name__ == "__main__":
    unittest.main()
