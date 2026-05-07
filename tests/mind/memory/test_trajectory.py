"""Tests for ``quantmind.mind.memory._trajectory``."""

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quantmind.mind.memory._trajectory import (
    RunRecord,
    generate_run_id,
    write_run_record,
)


def _fixture_record(run_id: str) -> RunRecord:
    started = datetime(2026, 5, 8, 12, 30, 45, 123000, tzinfo=timezone.utc)
    ended = datetime(2026, 5, 8, 12, 30, 50, 456000, tzinfo=timezone.utc)
    return RunRecord(
        schema_version=1,
        run_id=run_id,
        workflow_name="quantmind.paper_flow",
        trace_id="trace_abc",
        started_at=started,
        ended_at=ended,
        duration_seconds=5.333,
        agent={
            "name": "paper_extractor",
            "model": "gpt-4o",
            "instructions_hash": "abc",
        },
        llm_calls=[
            {
                "tokens_in": 10,
                "tokens_out": 5,
                "duration_s": 0.5,
                "model": "gpt-4o",
            }
        ],
        tool_calls=[],
        memory_ops={"files_read": [], "files_written": []},
        tokens_total={"input": 10, "output": 5},
        cost_estimate_usd=0.0,
        input_summary="hello",
        output_summary="world",
        error=None,
    )


class GenerateRunIdTests(unittest.TestCase):
    def test_format_matches_documented_shape(self) -> None:
        now = datetime(2026, 5, 8, 12, 30, 45, 123456, tzinfo=timezone.utc)
        run_id = generate_run_id(now)
        self.assertRegex(run_id, r"^20260508T123045123[0-9a-z]{3}$")

    def test_different_calls_produce_different_ids(self) -> None:
        now = datetime(2026, 5, 8, 12, 30, 45, 123000, tzinfo=timezone.utc)
        ids = {generate_run_id(now) for _ in range(50)}
        self.assertGreater(len(ids), 40)


class WriteRunRecordTests(unittest.IsolatedAsyncioTestCase):
    async def test_writes_atomic_per_run_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            mem_dir = Path(raw_dir)
            (mem_dir / "runs").mkdir()
            record = _fixture_record("rid001abc")
            await write_run_record(mem_dir, record, archive_lock=asyncio.Lock())
            target = mem_dir / "runs" / "rid001abc.json"
            self.assertTrue(target.exists())
            tmp = mem_dir / "runs" / "rid001abc.json.tmp"
            self.assertFalse(tmp.exists())
            payload = json.loads(target.read_text())
            self.assertEqual(payload["run_id"], "rid001abc")
            self.assertEqual(
                payload["started_at"], "2026-05-08T12:30:45.123000Z"
            )

    async def test_appends_one_line_to_runs_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            mem_dir = Path(raw_dir)
            (mem_dir / "runs").mkdir()
            lock = asyncio.Lock()
            await write_run_record(
                mem_dir, _fixture_record("a01abc"), archive_lock=lock
            )
            await write_run_record(
                mem_dir, _fixture_record("b02def"), archive_lock=lock
            )
            lines = (mem_dir / "runs.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            ids = [json.loads(line)["run_id"] for line in lines]
            self.assertEqual(ids, ["a01abc", "b02def"])

    async def test_concurrent_writes_serialise_under_lock(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            mem_dir = Path(raw_dir)
            (mem_dir / "runs").mkdir()
            lock = asyncio.Lock()
            await asyncio.gather(
                write_run_record(
                    mem_dir,
                    _fixture_record("c01abc"),
                    archive_lock=lock,
                ),
                write_run_record(
                    mem_dir,
                    _fixture_record("c02def"),
                    archive_lock=lock,
                ),
            )
            lines = (mem_dir / "runs.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
