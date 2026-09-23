"""Micro-batch observable ``extract -> transform -> load`` execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, Awaitable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast

from quantmind.etl._record import (
    JsonScalar,
    PipelineContext,
    RunRecord,
    copy_scalar_mapping,
    validate_dry_run,
)

InputT = TypeVar("InputT")
ExtractedBatchT = TypeVar("ExtractedBatchT")
TransformedBatchT = TypeVar("TransformedBatchT")
_StageInputT = TypeVar("_StageInputT", contravariant=True)
_StageOutputT = TypeVar("_StageOutputT", covariant=True)
_BatchExtractInputT = TypeVar("_BatchExtractInputT", contravariant=True)
_BatchExtractOutputT = TypeVar("_BatchExtractOutputT", covariant=True)
_BatchLoadInputT = TypeVar("_BatchLoadInputT", contravariant=True)

CountDeltas: TypeAlias = Mapping[str, int] | None


class _BatchExtractCallable(
    Protocol[_BatchExtractInputT, _BatchExtractOutputT]
):
    def __call__(
        self,
        value: _BatchExtractInputT,
        /,
        *,
        ctx: PipelineContext,
    ) -> AsyncIterable[_BatchExtractOutputT]: ...


class _StageCallable(Protocol[_StageInputT, _StageOutputT]):
    def __call__(
        self,
        value: _StageInputT,
        /,
        *,
        ctx: PipelineContext,
    ) -> Awaitable[_StageOutputT]: ...


class _BatchLoadCallable(Protocol[_BatchLoadInputT]):
    def __call__(
        self,
        value: _BatchLoadInputT,
        /,
        *,
        ctx: PipelineContext,
    ) -> Awaitable[CountDeltas]: ...


@dataclass(slots=True)
class _BatchSnapshotState:
    index: int | None
    completed: int
    total: int | None

    def as_json(self) -> dict[str, object]:
        return {
            "index": self.index,
            "completed": self.completed,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class BatchRunSummary:
    """Bounded-memory summary returned by a successful batch ETL run."""

    completed_batches: int
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if type(self.completed_batches) is not int:
            raise TypeError("completed_batches must be an integer")
        if self.completed_batches < 0:
            raise ValueError("completed_batches must be >= 0")
        object.__setattr__(
            self,
            "counts",
            MappingProxyType(_copy_count_mapping(self.counts, "counts")),
        )


class BatchETLPipeline(Generic[InputT, ExtractedBatchT, TransformedBatchT]):
    """Bind async callables into a strict serial micro-batch ETL operation.

    Every authored batch ETL should honor the required ``dry_run`` run option:
    callers pass ``create_run(..., dry_run=dry_run)`` once, and each extract,
    transform, and load scope reads the same immutable ``ctx.dry_run`` value.
    Dry-run still pulls, transforms, and loads every batch serially; the
    authored ``load`` must validate or plan without persistent business writes,
    including staging/checkpoint writes, and return ``planned_*`` count deltas
    instead of delivered counts. AI processing may still run and incur cost or
    rate usage.
    """

    def __init__(
        self,
        name: str,
        *,
        extract: _BatchExtractCallable[InputT, ExtractedBatchT],
        transform: _StageCallable[ExtractedBatchT, TransformedBatchT],
        load: _BatchLoadCallable[TransformedBatchT],
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("pipeline name must be a non-empty string")
        if (
            not callable(extract)
            or not callable(transform)
            or not callable(load)
        ):
            raise TypeError("extract, transform, and load must be callable")
        self._name = name
        self._extract = extract
        self._transform = transform
        self._load = load

    @property
    def name(self) -> str:
        """The stable name written to each local run record."""
        return self._name

    def create_run(
        self,
        source: InputT,
        *,
        dry_run: bool,
        config_summary: Mapping[str, JsonScalar] | None = None,
        run_root: Path | None = None,
        total_batches: int | None = None,
    ) -> BatchPipelineRun[ExtractedBatchT, TransformedBatchT]:
        """Create a one-shot micro-batch run.

        ``dry_run`` is required because it changes side-effect semantics and is
        written as a top-level field from the initial ``created`` snapshot.
        Do not put ``dry_run`` in ``config_summary``. Dry-run runs the complete
        batch stage shape, including ``load`` for every yielded batch; stage
        authors read ``ctx.dry_run`` and forward it to the dependency that owns
        mutation decisions. In dry-run, those dependencies must not create
        persistent business mutations, including staging/checkpoint writes or
        final delivery, and count deltas should use ``planned_*`` names. AI
        processing is allowed and may incur cost.

        Args:
            source: Runtime input retained in memory until execution.
            dry_run: Explicit run mode. ``True`` plans and validates without
                persistent business delivery; ``False`` allows authored
                mutation boundaries to commit.
            config_summary: Explicit allowlist of safe JSON-scalar settings.
            run_root: Directory that will contain the run directory. The
                default is ``<cwd>/.quant-mind/etl-pipeline-runs``.
            total_batches: Optional known batch count. Unknown is recorded as
                ``null`` until the extractor is exhausted.

        Returns:
            A one-shot batch run whose ``status_file`` already exists in
            ``created`` state.
        """
        root = (
            Path.cwd() / ".quant-mind" / "etl-pipeline-runs"
            if run_root is None
            else Path(run_root)
        ).resolve()
        return BatchPipelineRun._create(
            pipeline_name=self._name,
            source=source,
            extract=self._extract,
            transform=self._transform,
            load=self._load,
            dry_run=validate_dry_run(dry_run),
            config_summary=copy_scalar_mapping(
                config_summary, field_name="config_summary"
            ),
            run_root=root,
            total_batches=_validate_total_batches(total_batches),
        )


class BatchPipelineRun(Generic[ExtractedBatchT, TransformedBatchT]):
    """One-shot execution handle for a serial micro-batch ETL run."""

    def __init__(
        self,
        *,
        record: RunRecord,
        batch: _BatchSnapshotState,
        source: object,
        extract: _BatchExtractCallable[Any, ExtractedBatchT],
        transform: _StageCallable[ExtractedBatchT, TransformedBatchT],
        load: _BatchLoadCallable[TransformedBatchT],
    ) -> None:
        self._record = record
        self._batch = batch
        self._source = source
        self._extract = extract
        self._transform = transform
        self._load = load
        self._counts: dict[str, int] = {}
        self._executed = False

    @classmethod
    def _create(
        cls,
        *,
        pipeline_name: str,
        source: object,
        extract: _BatchExtractCallable[Any, ExtractedBatchT],
        transform: _StageCallable[ExtractedBatchT, TransformedBatchT],
        load: _BatchLoadCallable[TransformedBatchT],
        dry_run: bool,
        config_summary: dict[str, JsonScalar],
        run_root: Path,
        total_batches: int | None,
    ) -> BatchPipelineRun[ExtractedBatchT, TransformedBatchT]:
        batch = _BatchSnapshotState(
            index=None, completed=0, total=total_batches
        )
        record = RunRecord.create(
            pipeline_name=pipeline_name,
            dry_run=dry_run,
            config_summary=config_summary,
            run_root=run_root,
            snapshot_schema="quantmind.etl.batch-run/v1",
            event_schema="quantmind.etl.batch-event/v1",
            snapshot_extra=lambda: {"batch": batch.as_json()},
        )
        return cls(
            record=record,
            batch=batch,
            source=source,
            extract=extract,
            transform=transform,
            load=load,
        )

    @property
    def id(self) -> str:
        """The unique local run ID."""
        return self._record.id

    @property
    def dry_run(self) -> bool:
        """Whether this run plans and validates without business delivery."""
        return self._record.dry_run

    @property
    def status_file(self) -> Path:
        """Absolute path to the latest atomic ``run.json`` snapshot."""
        return self._record.status_file

    @property
    def events_file(self) -> Path:
        """Absolute path to the sparse lifecycle ``events.jsonl`` record."""
        return self._record.events_file

    def receipt(self) -> str:
        """Return the run ID and absolute status path as one JSON line."""
        return json.dumps(
            {
                "event": "etl_batch_run_created",
                "run_id": self._record.id,
                "dry_run": self._record.dry_run,
                "status_file": str(self._record.status_file),
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    async def execute(self) -> BatchRunSummary:
        """Run every extracted batch through ``transform`` then ``load``.

        The executor is strictly serial: it never starts batch ``N+1`` before
        batch ``N`` has loaded successfully. Cancellation and stage exceptions
        are recorded and the original exception is re-raised. After a yielded
        batch fails or is cancelled, user extractor ``aclose()`` is not started
        because an async cleanup coroutine cannot be forcibly bounded on the
        active event loop.
        ``dry_run=True`` does not skip ``load``; the authored load uses
        ``ctx.dry_run`` to validate or plan without committing, and reports
        planned count deltas honestly.
        """
        if self._executed:
            raise RuntimeError(
                "BatchPipelineRun.execute() can only be called once"
            )
        self._executed = True
        self._record.start_running()
        try:
            self._record.append_event("run_started")
        except Exception:
            pass

        try:
            extract_context = self._record.new_inactive_context(stage="extract")
            batches = self._extract(self._source, ctx=extract_context)
            try:
                iterator = batches.__aiter__()
            except AttributeError as exc:
                raise TypeError("extract must return an AsyncIterable") from exc

            await self._execute_batches(iterator, extract_context)
        except asyncio.CancelledError:
            try:
                self._record.finish("cancelled", "run_cancelled")
            except Exception:
                pass
            raise
        except Exception as exc:
            self._record.set_error(exc)
            try:
                self._record.finish("failed", "run_failed")
            except Exception:
                pass
            raise

        self._batch.index = None
        self._record.stage = "load" if self._batch.completed else None
        self._record.clear_progress()
        self._record.finish("succeeded", "run_succeeded")
        return BatchRunSummary(
            completed_batches=self._batch.completed,
            counts=self._counts,
        )

    async def _execute_batches(
        self,
        iterator: object,
        extract_context: PipelineContext,
    ) -> None:
        while True:
            batch_index = self._batch.completed + 1
            self._batch.index = batch_index
            context = self._record.start_context(
                "extract", batch_index=batch_index, context=extract_context
            )
            try:
                try:
                    extracted = await anext(cast(Any, iterator))
                except StopAsyncIteration:
                    context._close()
                    self._record.complete_context(context)
                    if (
                        self._batch.total is not None
                        and self._batch.completed != self._batch.total
                    ):
                        raise ValueError(
                            "completed batches must equal total_batches"
                        ) from None
                    self._batch.index = None
                    return
            finally:
                context._close()

            self._record.complete_context(context)
            if (
                self._batch.total is not None
                and batch_index > self._batch.total
            ):
                raise ValueError(
                    "extract yielded more batches than total_batches"
                )

            transformed = await self._transform_batch(extracted, batch_index)
            counts = await self._load_batch(transformed, batch_index)
            _merge_counts(self._counts, counts)
            self._batch.completed += 1
            self._record.write_snapshot()
            self._record.append_event(
                "batch_completed",
                stage="load",
                batch=self._batch.as_json(),
                counts=counts,
            )

    async def _transform_batch(
        self, extracted: ExtractedBatchT, batch_index: int
    ) -> TransformedBatchT:
        context = self._record.start_context(
            "transform", batch_index=batch_index
        )
        try:
            return await self._transform(extracted, ctx=context)
        finally:
            context._close()
            if self._record.active_context is context:
                self._record.complete_context(context)

    async def _load_batch(
        self, transformed: TransformedBatchT, batch_index: int
    ) -> dict[str, int]:
        context = self._record.start_context("load", batch_index=batch_index)
        try:
            result = await self._load(transformed, ctx=context)
            return _copy_count_deltas(result)
        finally:
            context._close()
            if self._record.active_context is context:
                self._record.complete_context(context)


def _validate_total_batches(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("total_batches must be an integer or None")
    if value < 0:
        raise ValueError("total_batches must be >= 0")
    return value


def _copy_count_deltas(value: CountDeltas) -> dict[str, int]:
    if value is None:
        return {}
    return _copy_count_mapping(value, "load counts")


def _copy_count_mapping(
    value: Mapping[str, int],
    field_name: str,
) -> dict[str, int]:
    copied: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
        if type(item) is not int:
            raise TypeError(f"{field_name}[{key!r}] must be an integer")
        if item < 0:
            raise ValueError(f"{field_name}[{key!r}] must be >= 0")
        copied[key] = item
    return copied


def _merge_counts(
    totals: dict[str, int],
    deltas: Mapping[str, int],
) -> None:
    for key, value in deltas.items():
        totals[key] = totals.get(key, 0) + value
