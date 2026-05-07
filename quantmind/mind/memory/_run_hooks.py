"""``MemoryRunHooks`` — per-run lifecycle accumulator + ``persist()``.

Constructed fresh per ``paper_flow`` invocation by
``FilesystemMemory.run_hooks()``. Each lifecycle method accumulates
metrics; the runner calls ``persist()`` in a ``finally`` block so
runs that fail still produce a trajectory record (with ``error``
set to ``str(exc)``).
"""

import asyncio
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import RunHooks

from quantmind.mind.memory._trajectory import (
    RunRecord,
    generate_run_id,
    write_run_record,
)

_SUMMARY_LIMIT = 500
_TRUNC_SUFFIX = "... [truncated]"


def _truncate(s: str, limit: int = _SUMMARY_LIMIT) -> str:
    if len(s) <= limit:
        return s
    head = limit - len(_TRUNC_SUFFIX)
    return s[:head] + _TRUNC_SUFFIX


def _instructions_hash(instructions: str | None) -> str:
    if not instructions:
        return ""
    return hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:16]


def _safe_repr(obj: Any) -> str:
    if obj is None:
        return ""
    dump = getattr(obj, "model_dump_json", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


class MemoryRunHooks(RunHooks[Any]):
    """Per-run lifecycle accumulator that persists a ``RunRecord``.

    State lives on the instance and is written exactly once via
    ``persist`` — typically called by the runner in a ``finally``
    block so failed runs still archive.
    """

    def __init__(self, *, memory_dir: Path, archive_lock: asyncio.Lock) -> None:
        self._memory_dir = memory_dir
        self._archive_lock = archive_lock

        self._started_at: datetime | None = None
        self._ended_at: datetime | None = None
        self._agent_name: str = ""
        self._agent_model: str = ""
        self._instructions_hash: str = ""
        self._output_summary: str = ""
        self._llm_calls: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []

        self._llm_timer_start: float | None = None
        self._tool_timer_starts: dict[int, float] = {}

    async def on_agent_start(self, ctx: Any, agent: Any) -> None:
        self._started_at = datetime.now(timezone.utc)
        self._agent_name = str(getattr(agent, "name", "") or "")
        self._agent_model = str(getattr(agent, "model", "") or "")
        self._instructions_hash = _instructions_hash(
            getattr(agent, "instructions", None)
        )

    async def on_llm_start(self, *_: Any, **__: Any) -> None:
        self._llm_timer_start = time.monotonic()

    async def on_llm_end(self, ctx: Any, agent: Any, response: Any) -> None:
        duration = (
            (time.monotonic() - self._llm_timer_start)
            if self._llm_timer_start is not None
            else 0.0
        )
        self._llm_timer_start = None
        usage = getattr(response, "usage", None)
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        model = str(getattr(agent, "model", "") or "")
        self._llm_calls.append(
            {
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "duration_s": round(duration, 4),
                "model": model,
            }
        )

    async def on_tool_start(self, ctx: Any, agent: Any, tool: Any) -> None:
        self._tool_timer_starts[id(tool)] = time.monotonic()

    async def on_tool_end(
        self, ctx: Any, agent: Any, tool: Any, result: Any
    ) -> None:
        start = self._tool_timer_starts.pop(id(tool), None)
        duration = (time.monotonic() - start) if start is not None else 0.0
        name = str(getattr(tool, "name", "") or "")
        self._tool_calls.append(
            {
                "name": name,
                "args": (_truncate(str(result)) if result is not None else ""),
                "duration_s": round(duration, 4),
            }
        )

    async def on_agent_end(self, ctx: Any, agent: Any, output: Any) -> None:
        self._ended_at = datetime.now(timezone.utc)
        self._output_summary = _truncate(_safe_repr(output))

    async def persist(
        self,
        *,
        workflow_name: str,
        result: Any,
        error: BaseException | None,
        input_payload: Any,
    ) -> None:
        """Build a ``RunRecord`` from accumulated state and write it.

        Called by the runner in ``finally`` so failed runs archive too.
        """
        ended = self._ended_at or datetime.now(timezone.utc)
        started = self._started_at or ended
        tokens_total = {
            "input": sum(c["tokens_in"] for c in self._llm_calls),
            "output": sum(c["tokens_out"] for c in self._llm_calls),
        }
        record = RunRecord(
            schema_version=1,
            run_id=generate_run_id(started),
            workflow_name=workflow_name,
            trace_id=(getattr(result, "trace_id", None) if result else None),
            started_at=started,
            ended_at=ended,
            duration_seconds=round((ended - started).total_seconds(), 4),
            agent={
                "name": self._agent_name,
                "model": self._agent_model,
                "instructions_hash": self._instructions_hash,
            },
            llm_calls=list(self._llm_calls),
            tool_calls=list(self._tool_calls),
            memory_ops={"files_read": [], "files_written": []},
            tokens_total=tokens_total,
            cost_estimate_usd=0.0,
            input_summary=_truncate(_safe_repr(input_payload)),
            output_summary=self._output_summary,
            error=str(error) if error is not None else None,
        )
        await write_run_record(
            self._memory_dir, record, archive_lock=self._archive_lock
        )
