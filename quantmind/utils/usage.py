"""Per-run token, timing, and step usage, read from the Agents SDK's traces.

A flow run makes several model calls (a paper summary fans out ``N`` research
agents and one reducer, each its own ``Runner.run``), and the Agents SDK already
records every one as a typed span carrying token usage and ISO start/end
timestamps. This module collects those spans instead of hand-rolling a second
accounting path, and reports one ``RunUsage`` per flow run.

Usage:

    from quantmind.utils.usage import usage_scope

    with usage_scope("quantmind.paper") as run:
        tree = await PaperFlow(cfg).build(input)

    print(run.usage.total_tokens, run.usage.requests, run.usage.wall_seconds)
    for step in run.usage.steps:
        print(step.label, step.model, step.total_tokens, step.duration_seconds)

``usage_scope`` opens one SDK ``trace`` (a minted ``trace_id``) so every
``Runner.run`` inside nests into a single tree, then reads and evicts that
trace's spans on exit. It only *adds* a local ``TracingProcessor`` via
``add_trace_processor`` — it never replaces the SDK's processors and never
touches the default backend exporter, so whether spans also reach any external
dashboard stays the caller's own global tracing configuration.

This is observation only: it reads usage after the fact and never inspects it to
cap, price, or steer a run. Tokens/time/steps only — no cost, no budget.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agents import add_trace_processor, gen_trace_id, trace
from agents.tracing import TracingProcessor

# ``span_data.type`` values for a single model call — the leaf spans that carry
# one request's token usage. ``task`` / ``turn`` spans also expose ``usage`` but
# aggregate their children, so summing them would double-count.
_MODEL_SPAN_TYPES = frozenset({"generation", "response"})
_AGENT_SPAN_TYPE = "agent"


@dataclass(frozen=True, slots=True)
class UsageStep:
    """One model call within a run — a degenerate span kept for optimization.

    ``label`` is the enclosing agent's name (e.g. ``paper_chunk_researcher``),
    resolved from the span tree. ``started_at`` / ``ended_at`` are the SDK's ISO
    timestamps; ``span_id`` / ``parent_id`` let a caller rebuild the tree for a
    waterfall.
    """

    label: str
    model: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    started_at: str | None
    ended_at: str | None
    duration_seconds: float
    span_id: str
    parent_id: str | None


@dataclass(frozen=True, slots=True)
class RunUsage:
    """Aggregate token/time/step usage for one flow run.

    ``requests`` is the number of model-call spans — the "llm-steps" count.
    ``wall_seconds`` is true end-to-end latency (last end minus first start);
    ``busy_seconds`` is the sum of per-step durations. Their gap is how much the
    fan-out actually overlapped — the signal for tuning concurrency.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    requests: int
    wall_seconds: float
    busy_seconds: float
    steps: tuple[UsageStep, ...]

    @classmethod
    def empty(cls) -> RunUsage:
        """A zeroed ``RunUsage`` (a run with no recorded model calls)."""
        return cls(0, 0, 0, 0, 0.0, 0.0, ())


@dataclass
class UsageScope:
    """Handle yielded by ``usage_scope``; ``usage`` is filled when the block exits.

    Read ``usage`` *after* the ``with`` block: spans are aggregated on exit, so
    inside the block it is still the empty default.
    """

    trace_id: str
    usage: RunUsage = field(default_factory=RunUsage.empty)


class SpanCollector(TracingProcessor):
    """Process-global sink that buckets finished spans by ``trace_id``.

    Bucketing by ``trace_id`` is what makes concurrent scopes and ``batch_run``
    fan-out safe: each scope only ever reads and evicts its own trace.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_trace: dict[str, list[Any]] = defaultdict(list)

    def on_trace_start(self, trace: Any) -> None:  # noqa: D102 - no-op sink
        return None

    def on_trace_end(self, trace: Any) -> None:  # noqa: D102 - no-op sink
        return None

    def on_span_start(self, span: Any) -> None:  # noqa: D102 - no-op sink
        return None

    def on_span_end(self, span: Any) -> None:
        """Store the finished span under its trace bucket (thread-safe)."""
        with self._lock:
            self._by_trace[span.trace_id].append(span)

    def shutdown(self) -> None:  # noqa: D102 - nothing to flush
        return None

    def force_flush(self) -> None:  # noqa: D102 - synchronous, nothing queued
        return None

    def pop(self, trace_id: str) -> list[Any]:
        """Return and remove every span collected for ``trace_id``."""
        with self._lock:
            return self._by_trace.pop(trace_id, [])


_collector: SpanCollector | None = None
_collector_lock = threading.Lock()


def _get_collector() -> SpanCollector:
    """Return the process-global collector, registering it once on first use.

    Registration is *additive* (``add_trace_processor``): the SDK's own
    processors and default exporter are left untouched.
    """
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                collector = SpanCollector()
                add_trace_processor(collector)
                _collector = collector
    return _collector


@contextmanager
def usage_scope(
    workflow_name: str = "quantmind",
    *,
    group_id: str | None = None,
) -> Iterator[UsageScope]:
    """Trace one flow run and expose its ``RunUsage`` after the block.

    Args:
        workflow_name: Trace name for this run (shown in any tracing UI).
        group_id: Optional grouping id to link several runs (e.g. one per input
            across a ``batch_run``) into one logical session.

    Yields:
        A ``UsageScope`` whose ``usage`` is populated when the block exits.
        Reading it inside the block returns the empty default; read it after.
    """
    collector = _get_collector()
    trace_id = gen_trace_id()
    scope = UsageScope(trace_id=trace_id)
    try:
        with trace(workflow_name, trace_id=trace_id, group_id=group_id):
            yield scope
    finally:
        # Aggregate on exit even when the block raised, so a failed run still
        # reports whatever model calls it managed to make before failing.
        scope.usage = _aggregate(collector.pop(trace_id))


def _aggregate(spans: list[Any]) -> RunUsage:
    """Reduce one trace's spans into a ``RunUsage`` (leaf model spans only)."""
    by_id: dict[str, Any] = {s.span_id: s for s in spans}
    steps: list[UsageStep] = []
    for span in spans:
        span_data = span.span_data
        if getattr(span_data, "type", None) not in _MODEL_SPAN_TYPES:
            continue
        tokens = _extract_usage(span_data) or (0, 0, 0)
        steps.append(
            UsageStep(
                label=_resolve_label(span, by_id),
                model=_model_of(span_data),
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                total_tokens=tokens[2],
                started_at=span.started_at,
                ended_at=span.ended_at,
                duration_seconds=_duration(span.started_at, span.ended_at),
                span_id=span.span_id,
                parent_id=span.parent_id,
            )
        )
    steps.sort(key=lambda step: step.started_at or "")
    return RunUsage(
        input_tokens=sum(s.input_tokens for s in steps),
        output_tokens=sum(s.output_tokens for s in steps),
        total_tokens=sum(s.total_tokens for s in steps),
        requests=len(steps),
        wall_seconds=_wall_seconds(steps),
        busy_seconds=round(sum(s.duration_seconds for s in steps), 6),
        steps=tuple(steps),
    )


def _extract_usage(span_data: Any) -> tuple[int, int, int] | None:
    """Pull (input, output, total) tokens from a model span's usage.

    The SDK normalizes chat-completions ``prompt``/``completion`` to
    ``input_tokens``/``output_tokens`` in the span usage dict; a response span
    may instead expose it on ``response.usage``. Both are handled.
    """
    usage = getattr(span_data, "usage", None)
    if isinstance(usage, dict) and usage:
        tokens_in = _as_int(
            usage.get("input_tokens", usage.get("prompt_tokens"))
        )
        tokens_out = _as_int(
            usage.get("output_tokens", usage.get("completion_tokens"))
        )
        total = _as_int(usage.get("total_tokens")) or (tokens_in + tokens_out)
        return tokens_in, tokens_out, total
    response_usage = getattr(
        getattr(span_data, "response", None), "usage", None
    )
    if response_usage is not None:
        tokens_in = _as_int(getattr(response_usage, "input_tokens", None))
        tokens_out = _as_int(getattr(response_usage, "output_tokens", None))
        total = _as_int(getattr(response_usage, "total_tokens", None)) or (
            tokens_in + tokens_out
        )
        return tokens_in, tokens_out, total
    return None


def _model_of(span_data: Any) -> str | None:
    """Model name — direct on a generation span, on ``response`` for a response."""
    model = getattr(span_data, "model", None)
    if model:
        return str(model)
    response_model = getattr(
        getattr(span_data, "response", None), "model", None
    )
    return str(response_model) if response_model else None


def _resolve_label(span: Any, by_id: dict[str, Any]) -> str:
    """Nearest enclosing agent name, walking parents; else the span's own type."""
    current = span
    seen: set[str] = set()
    while True:
        parent_id = getattr(current, "parent_id", None)
        if parent_id is None or parent_id in seen:
            break
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            break
        if getattr(parent.span_data, "type", None) == _AGENT_SPAN_TYPE:
            return str(getattr(parent.span_data, "name", "") or "")
        current = parent
    return str(getattr(span.span_data, "type", "") or "")


def _wall_seconds(steps: list[UsageStep]) -> float:
    """End-to-end latency: last ``ended_at`` minus first ``started_at``."""
    starts = [dt for dt in (_parse_iso(s.started_at) for s in steps) if dt]
    ends = [dt for dt in (_parse_iso(s.ended_at) for s in steps) if dt]
    if not starts or not ends:
        return 0.0
    return round(max(0.0, (max(ends) - min(starts)).total_seconds()), 6)


def _duration(started_at: str | None, ended_at: str | None) -> float:
    start, end = _parse_iso(started_at), _parse_iso(ended_at)
    if start is None or end is None:
        return 0.0
    return round(max(0.0, (end - start).total_seconds()), 6)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0
