"""Batch runner — fan a flow function out over many inputs.

`batch_run` is the single concurrency primitive QuantMind ships in MVP.
It does NOT support `memory=`; for memory-accumulating workflows users
write a serial `for` loop themselves (design doc §4.3.5). This keeps the
batch path stateless and free of cross-run race hazards.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from quantmind.configs import BaseFlowCfg, BaseInput

InputT = TypeVar("InputT", bound=BaseInput)
OutputT = TypeVar("OutputT")


@dataclass(slots=True)
class BatchResult(Generic[OutputT]):
    """Aggregate result of running a flow over many inputs.

    ``results[i]`` is the output for ``inputs[i]`` or ``None`` if that
    input failed. ``errors`` carries ``(index, exception)`` for every
    failure, sorted by index. ``successes`` and ``failures`` are
    convenience views derived from these primary fields.
    """

    total: int
    success_count: int
    failure_count: int
    results: list[OutputT | None]
    errors: list[tuple[int, Exception]]
    duration_seconds: float
    tokens_total: dict[str, int] = field(default_factory=dict)
    cost_estimate_usd: float = 0.0

    @property
    def successes(self) -> list[tuple[int, OutputT]]:
        """``(index, result)`` for every input that succeeded."""
        return [(i, r) for i, r in enumerate(self.results) if r is not None]

    @property
    def failures(self) -> list[tuple[int, Exception]]:
        """Alias for ``errors`` to mirror ``successes`` for symmetry."""
        return list(self.errors)


async def batch_run(
    flow_fn: Callable[..., Awaitable[OutputT]],
    inputs: list[InputT],
    *,
    cfg: BaseFlowCfg | None = None,
    concurrency: int = 4,
    on_error: Literal["raise", "skip"] = "skip",
    on_progress: Callable[[int, int], None] | None = None,
    **flow_kwargs: Any,
) -> BatchResult[OutputT]:
    """Run ``flow_fn`` over ``inputs`` with bounded concurrency.

    Args:
        flow_fn: Any flow function with signature
            ``(input, *, cfg, **kwargs) -> Awaitable[OutputT]``.
        inputs: Inputs to fan out over. Empty list returns an empty
            ``BatchResult`` immediately.
        cfg: Shared cfg forwarded to every call. ``None`` lets the flow
            use its own default.
        concurrency: Maximum number of in-flight calls. Must be ≥ 1.
        on_error: ``"raise"`` propagates the first failure (siblings get
            cancelled); ``"skip"`` records every failure into
            ``errors`` and returns the batch normally.
        on_progress: Called as ``on_progress(done, total)`` after every
            completion (success or failure). Must be cheap and
            non-blocking — callbacks are invoked synchronously inside
            the worker loop.
        **flow_kwargs: Forwarded verbatim to ``flow_fn``. ``memory=`` is
            **forbidden** in MVP; passing it raises ``ValueError``.

    Returns:
        ``BatchResult`` with ``results`` parallel to ``inputs`` (None for
        failures) and ``errors`` sorted by index.

    Raises:
        ValueError: If ``memory=`` is passed via ``flow_kwargs``, or if
            ``concurrency < 1``.
        Exception: Re-raised when ``on_error="raise"`` and any input
            fails. The exception is the first one raised by a worker;
            other workers may already be cancelled when this surfaces.
    """
    if "memory" in flow_kwargs:
        raise ValueError(
            "batch_run does not support `memory=` in MVP. For "
            "memory-accumulating workflows write a serial loop instead: "
            "`for inp in inputs: await flow_fn(inp, cfg=cfg, memory=memory)`. "
            "See design doc §4.3.5."
        )
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    sem = asyncio.Semaphore(concurrency)
    results: list[OutputT | None] = [None] * len(inputs)
    errors: list[tuple[int, Exception]] = []
    started = time.monotonic()
    done_counter = 0

    async def run_one(i: int, inp: InputT) -> None:
        nonlocal done_counter
        async with sem:
            try:
                results[i] = await flow_fn(inp, cfg=cfg, **flow_kwargs)
            except Exception as exc:
                errors.append((i, exc))
                if on_error == "raise":
                    raise
            finally:
                # asyncio is single-threaded; this increment + read +
                # callback all happen synchronously between await points.
                done_counter += 1
                if on_progress is not None:
                    on_progress(done_counter, len(inputs))

    # Same call shape for both modes — `run_one` swallows its own
    # exception when on_error="skip", and re-raises (cancelling siblings
    # via gather's default behaviour) when on_error="raise".
    await asyncio.gather(*(run_one(i, inp) for i, inp in enumerate(inputs)))

    return BatchResult(
        total=len(inputs),
        success_count=sum(1 for r in results if r is not None),
        failure_count=len(errors),
        results=results,
        errors=sorted(errors, key=lambda t: t[0]),
        duration_seconds=time.monotonic() - started,
    )
