"""Per-run usage accounting layered on the flows observability seam.

The Agents SDK computes token usage as ``RunResult.context_wrapper.usage``,
but ``run_with_observability`` returns only ``final_output`` and drops the
rest. Flows return domain objects, so ``batch_run`` never sees usage.

This module surfaces usage across that seam without changing any flow's
return type. ``run_with_observability`` calls ``record_usage`` after every
run; a caller opens a ``usage_scope`` around the work it wants measured, and
each run folds its usage into the active accumulator. ``asyncio`` copies the
context (and with it the accumulator *reference*) into every child task, so
usage from nested ``gather`` / ``wait_for`` fan-outs accumulates into the one
scope the caller opened. Mutation has no ``await`` points, so no lock is
needed on the single-threaded event loop.
"""

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PriceRate:
    """Per-token USD pricing for one model (rates are per one million tokens)."""

    input_usd_per_1m: float
    output_usd_per_1m: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """USD cost for the given input / output token counts."""
        return (
            input_tokens / 1_000_000 * self.input_usd_per_1m
            + output_tokens / 1_000_000 * self.output_usd_per_1m
        )


@dataclass(frozen=True, slots=True)
class UsageSummary:
    """Immutable token-usage snapshot returned to callers.

    ``cost_usd`` is ``None`` unless the caller supplied a price table; the
    library reports tokens and leaves per-model pricing to the caller.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None

    def as_tokens_dict(self) -> dict[str, int]:
        """Return the token counts as a plain dict (no cost)."""
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class _Accumulator:
    """Mutable running total; duck-types the SDK ``Usage`` object."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: object) -> None:
        """Fold one SDK ``Usage`` (or ``UsageSummary``) into the total."""
        self.requests += getattr(usage, "requests", 0)
        self.input_tokens += getattr(usage, "input_tokens", 0)
        self.output_tokens += getattr(usage, "output_tokens", 0)
        self.total_tokens += getattr(usage, "total_tokens", 0)

    def summary(self) -> UsageSummary:
        """Snapshot the running total as an immutable ``UsageSummary``."""
        return UsageSummary(
            requests=self.requests,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
        )


_usage_var: ContextVar[_Accumulator | None] = ContextVar(
    "quantmind_usage", default=None
)


@contextlib.contextmanager
def usage_scope() -> Iterator[_Accumulator]:
    """Accumulate usage from every run inside this context.

    Yields the accumulator; read ``.summary()`` after the block. Nested
    ``usage_scope`` calls each measure only their own runs.
    """
    accumulator = _Accumulator()
    token = _usage_var.set(accumulator)
    try:
        yield accumulator
    finally:
        _usage_var.reset(token)


def record_usage(usage: object) -> None:
    """Fold one run's SDK usage into the active scope (no-op if none)."""
    accumulator = _usage_var.get()
    if accumulator is not None:
        accumulator.add(usage)
