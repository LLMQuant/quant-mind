"""Trajectory archive: ``RunRecord`` schema + atomic writer.

Each ``Runner.run`` invocation produces one ``RunRecord`` that is
serialised to ``<memory_dir>/runs/<run_id>.json`` (atomic via
``os.replace``) and appended to ``<memory_dir>/runs.jsonl`` (one
JSON line per run, append-only).

Cross-process concurrency is undefined; an ``asyncio.Lock`` (held by
the calling ``FilesystemMemory``) serialises writes within a single
Python process.
"""

import asyncio
import json
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True, slots=True)
class RunRecord:
    """A single QuantMind flow run trajectory record."""

    schema_version: int
    run_id: str
    workflow_name: str
    trace_id: str | None
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    agent: dict[str, str]
    llm_calls: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    memory_ops: dict[str, list[str]]
    tokens_total: dict[str, int]
    cost_estimate_usd: float
    input_summary: str
    output_summary: str
    error: str | None


def generate_run_id(now: datetime) -> str:
    """Build ``YYYYMMDDTHHMMSSmmm`` (UTC) plus 3 base36 random chars."""
    stamp = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}"
    suffix = "".join(secrets.choice(_BASE36) for _ in range(3))
    return f"{stamp}{suffix}"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serialisable"
    )


def _serialise(record: RunRecord) -> str:
    return json.dumps(
        asdict(record),
        default=_to_jsonable,
        ensure_ascii=False,
    )


async def write_run_record(
    memory_dir: Path,
    record: RunRecord,
    *,
    archive_lock: asyncio.Lock,
) -> None:
    """Persist ``record`` atomically and append to ``runs.jsonl``.

    Atomicity:

    - The per-run JSON file is written to ``<run_id>.json.tmp`` and
      then ``os.replace``-d into place; same-FS rename is atomic on
      POSIX, preventing partial reads.
    - The ``runs.jsonl`` append is serialised across one Python
      process via ``archive_lock``; cross-process concurrency is
      unsupported (documented in ``FilesystemMemory``).
    """
    payload = _serialise(record)
    runs_dir = memory_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    final = runs_dir / f"{record.run_id}.json"
    tmp = runs_dir / f"{record.run_id}.json.tmp"
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, final)

    index = memory_dir / "runs.jsonl"
    async with archive_lock:
        with index.open("a", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")
