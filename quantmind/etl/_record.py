"""Shared local run recording mechanics for observable ETL runs."""

from __future__ import annotations

import contextvars
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias

JsonScalar: TypeAlias = None | bool | int | float | str
StageName: TypeAlias = Literal["extract", "transform", "load"]
RunState: TypeAlias = Literal[
    "created", "running", "succeeded", "failed", "cancelled"
]

_PROGRESS_EVENT_INTERVAL_SECONDS = 1.0
_MAX_ERROR_MESSAGE_LENGTH = 1000
_CURRENT_PROGRESS_SCOPE: contextvars.ContextVar[tuple[int, int] | None] = (
    contextvars.ContextVar("quantmind_etl_progress_scope", default=None)
)
_RESERVED_CONFIG_SUMMARY_DRY_RUN_MESSAGE = (
    "config_summary['dry_run'] is reserved; use create_run(..., dry_run=...)"
)


def _utc_now() -> str:
    """Return a compact UTC timestamp suitable for local run records."""
    value = datetime.now(timezone.utc).replace(microsecond=0)
    return value.isoformat().replace("+00:00", "Z")


def _monotonic_seconds() -> float:
    return time.monotonic()


def _new_run_id(timestamp: str) -> str:
    compact_timestamp = timestamp.replace("-", "").replace(":", "")
    return f"qmr_{compact_timestamp}_{uuid.uuid4().hex[:8]}"


def copy_scalar_mapping(
    value: Mapping[str, JsonScalar] | None,
    *,
    field_name: str,
) -> dict[str, JsonScalar]:
    """Copy and validate an explicit JSON-scalar allowlist."""
    if value is None:
        return {}

    copied: dict[str, JsonScalar] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
        if field_name == "config_summary" and key == "dry_run":
            raise ValueError(_RESERVED_CONFIG_SUMMARY_DRY_RUN_MESSAGE)
        if item is not None and not isinstance(item, (bool, int, float, str)):
            raise TypeError(f"{field_name}[{key!r}] must be a JSON scalar")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{field_name}[{key!r}] must be finite")
        copied[key] = item
    return copied


def validate_dry_run(value: bool) -> bool:
    """Validate the explicit run-level dry-run flag."""
    if type(value) is not bool:
        raise TypeError("dry_run must be a boolean")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                _json_safe_value(value),
                stream,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_json_line(path: Path, value: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        json.dump(
            _json_safe_value(value),
            stream,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()


def _json_safe_string(value: str) -> str:
    return value.encode("utf-8", errors="backslashreplace").decode("utf-8")


def _json_safe_value(value: object) -> object:
    if isinstance(value, str):
        return _json_safe_string(value)
    if isinstance(value, Mapping):
        return {
            _json_safe_string(key) if isinstance(key, str) else key: (
                _json_safe_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    return value


def _exception_message(exc: BaseException) -> str:
    try:
        message = str(exc)
    except Exception as message_error:
        message = (
            f"<unprintable {type(exc).__name__}: "
            f"{type(message_error).__name__}>"
        )
    return _json_safe_string(message)[:_MAX_ERROR_MESSAGE_LENGTH]


@dataclass(frozen=True, slots=True)
class Progress:
    """Validated stage-local progress snapshot."""

    completed: int
    total: int | None
    message: str | None
    metrics: dict[str, JsonScalar]

    def as_json(self) -> dict[str, object]:
        """Return the JSON representation written to snapshots and events."""
        return {
            "completed": self.completed,
            "total": self.total,
            "message": self.message,
            "metrics": self.metrics,
        }


@dataclass(frozen=True, slots=True)
class _PendingProgress:
    progress: Progress
    occurred_at: str


class PipelineContext:
    """Stage-local context used to read run mode and report completed work.

    ``completed`` is the amount of work that has finished, not work that was
    queued or discovered. It must strictly increase within one active
    ``(batch_index, stage)`` scope. Pass ``total=None`` while the total is
    unknown; once a scope reports a known total, later updates for that scope
    must keep reporting a known total.

    Every ETL pipeline should implement dry-run behavior. Callers choose it
    once with ``create_run(..., dry_run=dry_run)``; all whole-run stages and all
    batch-stage scopes then read the same immutable ``ctx.dry_run`` value.
    Stage authors pass that flag to the repository, gateway, publisher, or
    other capability that owns mutation decisions. Dry-run still executes every
    stage, including ``load``, but authored code must prevent persistent
    business mutations such as database writes, storage artifacts, staging or
    checkpoint files, publish/queue/webhook calls, and final delivery. Use
    ``planned_*`` names for dry-run progress metrics and count deltas. AI
    processing may still run and incur cost or rate usage; dry-run is not a
    zero-cost mode.
    """

    def __init__(
        self,
        *,
        run_id: str,
        dry_run: bool,
        stage: StageName,
        report: Callable[[PipelineContext, Progress], None],
        batch_index: int | None = None,
        active: bool = True,
    ) -> None:
        self._run_id = run_id
        self._dry_run = dry_run
        self._stage = stage
        self._batch_index = batch_index
        self._report = report
        self._previous: Progress | None = None
        self._active = False
        self._activation_epoch = 0
        self._scope_token: contextvars.Token[tuple[int, int] | None] | None = (
            None
        )
        if active:
            self._activate(stage=stage, batch_index=batch_index)

    @property
    def run_id(self) -> str:
        """The ID of the run that owns this stage."""
        return self._run_id

    @property
    def dry_run(self) -> bool:
        """Whether this run is planning and validating without delivery."""
        return self._dry_run

    @property
    def stage(self) -> str:
        """The fixed macro stage currently being executed."""
        return self._stage

    @property
    def batch_index(self) -> int | None:
        """The current 1-based batch index, or ``None`` for whole-run ETL."""
        return self._batch_index

    async def progress(
        self,
        completed: int,
        *,
        total: int | None = None,
        message: str | None = None,
        metrics: Mapping[str, JsonScalar] | None = None,
    ) -> None:
        """Record real completed work for the current stage.

        Args:
            completed: Finished work. It must be a non-negative integer and
                strictly increase within this active ``(batch, stage)`` scope.
            total: Known total work, or ``None`` while it is unknown.
            message: Optional short, safe progress description.
            metrics: Optional allowlisted JSON-scalar measurements.

        Raises:
            RuntimeError: If the stage has already ended.
            TypeError: If progress values have invalid types.
            ValueError: If progress moves backwards or contradicts ``total``.
        """
        if not self._active:
            raise RuntimeError(
                "progress cannot be reported after the stage ends"
            )
        if _CURRENT_PROGRESS_SCOPE.get() != (
            id(self),
            self._activation_epoch,
        ):
            raise RuntimeError(
                "progress can only be reported by the active stage scope"
            )
        if type(completed) is not int:
            raise TypeError("completed must be an integer")
        if completed < 0:
            raise ValueError("completed must be >= 0")
        if total is not None and type(total) is not int:
            raise TypeError("total must be an integer or None")
        if total is not None and total < completed:
            raise ValueError("total must be >= completed")
        if message is not None and not isinstance(message, str):
            raise TypeError("message must be a string or None")

        previous = self._previous
        if previous is not None and completed <= previous.completed:
            raise ValueError("completed must strictly increase within a stage")
        if (
            previous is not None
            and previous.total is not None
            and total is None
        ):
            raise ValueError("total cannot become unknown after it was known")

        progress = Progress(
            completed=completed,
            total=total,
            message=message,
            metrics=copy_scalar_mapping(metrics, field_name="metrics"),
        )
        self._report(self, progress)
        self._previous = progress

    def _activate(self, *, stage: StageName, batch_index: int | None) -> None:
        self._stage = stage
        self._batch_index = batch_index
        self._previous = None
        self._active = True
        self._activation_epoch += 1
        self._scope_token = _CURRENT_PROGRESS_SCOPE.set(
            (id(self), self._activation_epoch)
        )

    def _close(self) -> None:
        self._active = False
        token = self._scope_token
        self._scope_token = None
        if token is not None:
            _CURRENT_PROGRESS_SCOPE.reset(token)


class RunRecord:
    """Atomic snapshot and sparse event writer shared by ETL run shapes."""

    def __init__(
        self,
        *,
        run_id: str,
        pipeline_name: str,
        dry_run: bool,
        config_summary: dict[str, JsonScalar],
        run_directory: Path,
        created_at: str,
        snapshot_schema: str,
        event_schema: str,
        snapshot_extra: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        self.id = run_id
        self.pipeline_name = pipeline_name
        self.dry_run = dry_run
        self.config_summary = config_summary
        self.run_directory = run_directory
        self.status_file = run_directory / "run.json"
        self.events_file = run_directory / "events.jsonl"
        self.created_at = created_at
        self.updated_at = created_at
        self.snapshot_schema = snapshot_schema
        self.event_schema = event_schema
        self.snapshot_extra = snapshot_extra
        self.state: RunState = "created"
        self.stage: StageName | None = None
        self.pid: int | None = None
        self.progress: Progress | None = None
        self.error: dict[str, str] | None = None
        self.active_context: PipelineContext | None = None
        self._last_progress_event_at: float | None = None
        self._pending_progress: _PendingProgress | None = None

    @classmethod
    def create(
        cls,
        *,
        pipeline_name: str,
        dry_run: bool,
        config_summary: dict[str, JsonScalar],
        run_root: Path,
        snapshot_schema: str,
        event_schema: str,
        snapshot_extra: Callable[[], Mapping[str, object]] | None = None,
    ) -> RunRecord:
        """Create the run directory and write the initial snapshot."""
        created_at = _utc_now()
        run_id = _new_run_id(created_at)
        run_root.mkdir(parents=True, exist_ok=True)
        run_directory = run_root / run_id
        run_directory.mkdir(mode=0o700)
        events_file = run_directory / "events.jsonl"
        events_file.touch(exist_ok=False)

        record = cls(
            run_id=run_id,
            pipeline_name=pipeline_name,
            dry_run=dry_run,
            config_summary=config_summary,
            run_directory=run_directory,
            created_at=created_at,
            snapshot_schema=snapshot_schema,
            event_schema=event_schema,
            snapshot_extra=snapshot_extra,
        )
        record.write_snapshot()
        return record

    def start_running(self) -> None:
        """Mark the run as running and write the first live snapshot."""
        self.state = "running"
        self.pid = os.getpid()
        self.updated_at = _utc_now()
        self.write_snapshot()

    def new_inactive_context(
        self,
        *,
        stage: StageName,
        batch_index: int | None = None,
    ) -> PipelineContext:
        """Create a reusable inactive context for lazy async iterators."""
        return PipelineContext(
            run_id=self.id,
            dry_run=self.dry_run,
            stage=stage,
            batch_index=batch_index,
            report=self.record_progress,
            active=False,
        )

    def start_context(
        self,
        stage: StageName,
        *,
        batch_index: int | None = None,
        context: PipelineContext | None = None,
        event: str | None = None,
    ) -> PipelineContext:
        """Start a stage or batch-stage context and reset progress state."""
        if self.active_context is not None:
            raise RuntimeError("another stage context is already active")
        self.stage = stage
        self.progress = None
        self._pending_progress = None
        self._last_progress_event_at = None
        self.updated_at = _utc_now()

        if context is None:
            context = PipelineContext(
                run_id=self.id,
                dry_run=self.dry_run,
                stage=stage,
                batch_index=batch_index,
                report=self.record_progress,
            )
        else:
            context._activate(stage=stage, batch_index=batch_index)

        self.active_context = context
        self.write_snapshot()
        if event is not None:
            self.append_event(event, stage=stage)
        return context

    def complete_context(
        self,
        context: PipelineContext,
        *,
        event: str | None = None,
    ) -> None:
        """Flush progress and mark a stage or batch-stage context complete."""
        if self.active_context is not context:
            raise RuntimeError("stage context is no longer active")
        self.flush_pending_progress()
        self.updated_at = _utc_now()
        self.write_snapshot()
        if event is not None:
            self.append_event(event, stage=context.stage)
        self.active_context = None

    def record_progress(
        self, context: PipelineContext, progress: Progress
    ) -> None:
        """Merge progress into the snapshot and coalesce progress events."""
        if self.state != "running" or self.active_context is not context:
            raise RuntimeError(
                "progress can only be reported by the active stage"
            )
        occurred_at = _utc_now()
        self.progress = progress
        self.updated_at = occurred_at
        self.write_snapshot()

        now = _monotonic_seconds()
        last = self._last_progress_event_at
        if last is None or now - last >= _PROGRESS_EVENT_INTERVAL_SECONDS:
            self._append_progress_event(
                context, progress, occurred_at=occurred_at
            )
            self._last_progress_event_at = now
            self._pending_progress = None
        else:
            self._pending_progress = _PendingProgress(progress, occurred_at)

    def flush_pending_progress(self) -> None:
        """Append the last coalesced progress update, if one exists."""
        pending = self._pending_progress
        context = self.active_context
        if pending is None or context is None:
            return
        self._append_progress_event(
            context,
            pending.progress,
            occurred_at=pending.occurred_at,
        )
        self._pending_progress = None

    def set_error(self, exc: BaseException) -> None:
        """Store a bounded, JSON-safe error summary."""
        self.error = {
            "type": type(exc).__name__,
            "message": _exception_message(exc),
        }

    def clear_progress(self) -> None:
        """Clear progress before a terminal snapshot that changes scope."""
        self.progress = None
        self._pending_progress = None
        self._last_progress_event_at = None

    def finish(self, state: RunState, event: str) -> None:
        """Write the terminal snapshot and best-effort lifecycle event."""
        if self.active_context is not None:
            self.active_context._close()
        try:
            self.flush_pending_progress()
        except Exception:
            self._pending_progress = None
        self.active_context = None
        self.state = state
        self.updated_at = _utc_now()
        self.write_snapshot()
        try:
            if self.error is None:
                self.append_event(event, stage=self.stage)
            else:
                self.append_event(event, stage=self.stage, error=self.error)
        except Exception:
            pass

    def write_snapshot(self) -> None:
        """Atomically write the current ``run.json`` snapshot."""
        value: dict[str, object] = {
            "schema": self.snapshot_schema,
            "run_id": self.id,
            "pipeline": self.pipeline_name,
            "dry_run": self.dry_run,
            "state": self.state,
            "stage": self.stage,
            "pid": self.pid,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config_summary": self.config_summary,
            "progress": (
                None if self.progress is None else self.progress.as_json()
            ),
            "error": self.error,
        }
        if self.snapshot_extra is not None:
            value.update(self.snapshot_extra())
        _write_json_atomic(self.status_file, value)

    def append_event(
        self,
        event: str,
        *,
        stage: str | None = None,
        occurred_at: str | None = None,
        **fields: object,
    ) -> None:
        """Append one sparse JSON event."""
        value: dict[str, object] = {
            "schema": self.event_schema,
            "event": event,
            "run_id": self.id,
            "dry_run": self.dry_run,
            "occurred_at": occurred_at or _utc_now(),
        }
        if stage is not None:
            value["stage"] = stage
        value.update(fields)
        _append_json_line(self.events_file, value)

    def _append_progress_event(
        self,
        context: PipelineContext,
        progress: Progress,
        *,
        occurred_at: str,
    ) -> None:
        fields: dict[str, object] = {
            "progress": progress.as_json(),
        }
        if context.batch_index is not None:
            fields["batch_index"] = context.batch_index
        self.append_event(
            "stage_progress",
            stage=context.stage,
            occurred_at=occurred_at,
            **fields,
        )
