"""Async, observable-by-default ``extract -> transform -> load`` execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Mapping
from pathlib import Path
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast

from quantmind.etl._record import (
    JsonScalar,
    PipelineContext,
    RunRecord,
    StageName,
    copy_scalar_mapping,
    validate_dry_run,
)

InputT = TypeVar("InputT")
ExtractedT = TypeVar("ExtractedT")
TransformedT = TypeVar("TransformedT")
OutputT = TypeVar("OutputT")
_StageInputT = TypeVar("_StageInputT", contravariant=True)
_StageOutputT = TypeVar("_StageOutputT", covariant=True)


class _StageCallable(Protocol[_StageInputT, _StageOutputT]):
    def __call__(
        self,
        value: _StageInputT,
        /,
        *,
        ctx: PipelineContext,
    ) -> Awaitable[_StageOutputT]: ...


_Stage: TypeAlias = _StageCallable[Any, Any]


class ETLPipeline(Generic[InputT, ExtractedT, TransformedT, OutputT]):
    """Bind three async callables into one fixed, observable ETL operation.

    This class uses composition instead of an inheritance hierarchy. Coding
    agents implement the three stage functions and bind them once here; every
    call to :meth:`create_run` then owns independent run-specific state.
    Every authored ETL should honor the required ``dry_run`` run option:
    callers pass ``create_run(..., dry_run=dry_run)`` and stage functions read
    ``ctx.dry_run`` to pass the mode to the real mutation boundary. Dry-run
    still executes ``extract``, ``transform``, and ``load``; the authored
    ``load`` must validate or plan without persistent business writes,
    including staging/checkpoint writes, and return an honest planned result
    rather than a reference to a nonexistent delivery. AI processing may still
    run and incur cost or rate usage.
    """

    def __init__(
        self,
        name: str,
        *,
        extract: _StageCallable[InputT, ExtractedT],
        transform: _StageCallable[ExtractedT, TransformedT],
        load: _StageCallable[TransformedT, OutputT],
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
    ) -> PipelineRun[OutputT]:
        """Create a one-shot run and atomically write its initial snapshot.

        Only ``config_summary`` is persisted. The source, stage values, full
        configuration, and eventual result are kept out of the run record.
        ``dry_run`` is required because it changes side-effect semantics and is
        written as a top-level field from the initial ``created`` snapshot.
        Do not put ``dry_run`` in ``config_summary``.

        Dry-run runs the complete stage shape, including ``load``. Stage
        authors read ``ctx.dry_run`` and forward it to repositories, gateways,
        publishers, or other mutation-capable dependencies. In dry-run, those
        dependencies must not create persistent business mutations, including
        staging/checkpoint writes or final delivery; they may return planned
        paths, planned summaries, or ``None``. Use ``planned_*`` progress
        metrics in dry-run. AI processing is allowed and may incur cost.

        Args:
            source: Runtime input retained in memory until execution.
            dry_run: Explicit run mode. ``True`` plans and validates without
                persistent business delivery; ``False`` allows authored
                mutation boundaries to commit.
            config_summary: Explicit allowlist of safe JSON-scalar settings.
            run_root: Directory that will contain the run directory. The
                default is ``<cwd>/.quant-mind/etl-pipeline-runs``.

        Returns:
            A one-shot run whose ``status_file`` already exists in ``created``
            state.
        """
        root = (
            Path.cwd() / ".quant-mind" / "etl-pipeline-runs"
            if run_root is None
            else Path(run_root)
        ).resolve()
        return PipelineRun._create(
            pipeline_name=self._name,
            source=source,
            stages=(self._extract, self._transform, self._load),
            dry_run=validate_dry_run(dry_run),
            config_summary=copy_scalar_mapping(
                config_summary, field_name="config_summary"
            ),
            run_root=root,
        )


class PipelineRun(Generic[OutputT]):
    """One-shot execution handle with an atomically updated local snapshot."""

    def __init__(
        self,
        *,
        record: RunRecord,
        source: object,
        stages: tuple[_Stage, _Stage, _Stage],
    ) -> None:
        self._record = record
        self._source = source
        self._stages = stages
        self._executed = False

    @classmethod
    def _create(
        cls,
        *,
        pipeline_name: str,
        source: object,
        stages: tuple[_Stage, _Stage, _Stage],
        dry_run: bool,
        config_summary: dict[str, JsonScalar],
        run_root: Path,
    ) -> PipelineRun[Any]:
        record = RunRecord.create(
            pipeline_name=pipeline_name,
            dry_run=dry_run,
            config_summary=config_summary,
            run_root=run_root,
            snapshot_schema="quantmind.etl.run/v1",
            event_schema="quantmind.etl.event/v1",
        )
        return cls(
            record=record,
            source=source,
            stages=stages,
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
                "event": "etl_run_created",
                "run_id": self._record.id,
                "dry_run": self._record.dry_run,
                "status_file": str(self._record.status_file),
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    async def execute(self) -> OutputT:
        """Execute ``extract -> transform -> load`` exactly once.

        ``dry_run=True`` never changes the stage sequence; the stage functions
        are responsible for honoring ``ctx.dry_run`` at mutation boundaries and
        returning an honest planned value from ``load``.

        Cancellation is recorded as a terminal state on a best-effort basis,
        then the original :class:`asyncio.CancelledError` is re-raised.

        Raises:
            RuntimeError: If this handle has already been executed.
            Exception: Any exception raised by a stage after recording failure.
            asyncio.CancelledError: Re-raised after recording cancellation.
        """
        if self._executed:
            raise RuntimeError("PipelineRun.execute() can only be called once")
        self._executed = True
        self._record.start_running()
        try:
            self._record.append_event("run_started")
        except Exception:
            pass

        value: object = self._source
        try:
            for stage, operation in zip(
                cast(
                    tuple[StageName, StageName, StageName],
                    (
                        "extract",
                        "transform",
                        "load",
                    ),
                ),
                self._stages,
                strict=True,
            ):
                context = self._record.start_context(
                    stage, event="stage_started"
                )
                try:
                    value = await operation(value, ctx=context)
                finally:
                    context._close()
                self._record.complete_context(context, event="stage_completed")
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

        self._record.finish("succeeded", "run_succeeded")
        return cast(OutputT, value)
