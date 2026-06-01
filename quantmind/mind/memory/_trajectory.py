"""Trajectory archive: ``RunRecord`` schema + atomic writer.

Each ``Runner.run`` invocation produces one ``RunRecord`` that is
serialised to ``<memory_dir>/runs/<run_id>.json`` (atomic via
``os.replace``) and appended to ``<memory_dir>/runs.jsonl`` (one
JSON line per run, append-only).

Cross-process concurrency is undefined; an ``asyncio.Lock`` (held by
the calling ``FilesystemMemory``) serialises writes within a single
Python process. Both files and their parent directory entries are
``flush()``ed / ``os.fsync``ed before the writer returns so a crash
immediately after ``await write_run_record(...)`` does not leave the
per-run JSON or the ``runs.jsonl`` line lost in the kernel cache.
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
    """Build ``YYYYMMDDTHHMMSSmmm`` (UTC) plus 6 base36 random chars.

    6 chars give ``36**6 ≈ 2.2e9`` possibilities per millisecond. With
    realistic LLM-bound run rates (seconds per call) the birthday-paradox
    collision probability is negligible.
    """
    stamp = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}"
    suffix = "".join(secrets.choice(_BASE36) for _ in range(6))
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


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


async def write_run_record(
    memory_dir: Path,
    record: RunRecord,
    *,
    archive_lock: asyncio.Lock,
) -> None:
    """Persist ``record`` atomically and append to ``runs.jsonl``.

    Atomicity:

    - The per-run JSON file is written to a unique
      ``.<run_id>.<rand>.tmp`` and ``os.replace``-d into place — POSIX
      guarantees the rename is atomic on the same filesystem. The
      unique tmp name keeps two writers from clobbering each other's
      ``.tmp`` even if their ``run_id`` happens to collide.
    - The ``runs.jsonl`` append is serialised across one Python
      process via ``archive_lock``; cross-process concurrency is
      unsupported (documented in ``FilesystemMemory``).
    - Both files and their parent directory entries are explicitly
      ``flush()`` + ``os.fsync`` before the writer returns, so a crash
      directly after this coroutine awaits will not lose the run record.
    """
    payload = _serialise(record)
    runs_dir = memory_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    final = runs_dir / f"{record.run_id}.json"
    tmp = runs_dir / f".{record.run_id}.{secrets.token_hex(8)}.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    _fsync_directory(runs_dir)

    index = memory_dir / "runs.jsonl"
    async with archive_lock:
        with index.open("a", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        _fsync_directory(memory_dir)
