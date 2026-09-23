"""Tests for the observable micro-batch ETL scaffold."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import quantmind.etl._record as record_module
from quantmind.etl import BatchETLPipeline, PipelineContext


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _json_safe_text(value: str) -> str:
    return value.encode("utf-8", errors="backslashreplace").decode("utf-8")


class BatchPipelineRunTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.run_root = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    @staticmethod
    def _pipeline(
        *,
        extract=None,
        transform=None,
        load=None,
    ) -> BatchETLPipeline[list[int], int, int]:
        async def default_extract(source: list[int], *, ctx: PipelineContext):
            for item in source:
                yield item

        async def default_transform(batch: int, *, ctx: PipelineContext) -> int:
            return batch * 10

        async def default_load(
            batch: int, *, ctx: PipelineContext
        ) -> dict[str, int]:
            return {"batches": 1, "items": batch // 10}

        return BatchETLPipeline(
            "test-batch-pipeline",
            extract=extract or default_extract,
            transform=transform or default_transform,
            load=load or default_load,
        )

    async def test_create_run_writes_batch_snapshot_and_receipt(self) -> None:
        run = self._pipeline().create_run(
            [1, 2],
            dry_run=True,
            run_root=self.run_root,
            config_summary={"window_days": 7},
            total_batches=2,
        )

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["schema"], "quantmind.etl.batch-run/v1")
        self.assertTrue(snapshot["dry_run"])
        self.assertEqual(snapshot["state"], "created")
        self.assertIsNone(snapshot["stage"])
        self.assertEqual(
            snapshot["batch"], {"index": None, "completed": 0, "total": 2}
        )
        self.assertEqual(snapshot["config_summary"], {"window_days": 7})
        self.assertEqual(run.events_file.read_text(encoding="utf-8"), "")

        receipt = json.loads(run.receipt())
        self.assertEqual(receipt["event"], "etl_batch_run_created")
        self.assertEqual(receipt["run_id"], run.id)
        self.assertTrue(receipt["dry_run"])
        self.assertEqual(receipt["status_file"], str(run.status_file))
        self.assertTrue(Path(receipt["status_file"]).is_absolute())

    async def test_success_runs_each_batch_serially_with_honest_snapshot(
        self,
    ) -> None:
        run_holder = []
        calls: list[tuple[str, int | None, dict[str, object] | None]] = []
        generator_started = False

        async def extract(source: list[int], *, ctx: PipelineContext):
            nonlocal generator_started
            generator_started = True
            for item in source:
                snapshot = _read_json(run_holder[0].status_file)
                calls.append(
                    (
                        "extract",
                        ctx.batch_index,
                        snapshot["batch"],
                    )
                )
                self.assertEqual(snapshot["stage"], "extract")
                self.assertIsNone(snapshot["progress"])
                self.assertFalse(ctx.dry_run)
                yield item

        async def transform(batch: int, *, ctx: PipelineContext) -> int:
            snapshot = _read_json(run_holder[0].status_file)
            calls.append(("transform", ctx.batch_index, snapshot["batch"]))
            self.assertEqual(snapshot["stage"], "transform")
            self.assertIsNone(snapshot["progress"])
            self.assertFalse(ctx.dry_run)
            return batch * 10

        async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
            snapshot = _read_json(run_holder[0].status_file)
            calls.append(("load", ctx.batch_index, snapshot["batch"]))
            self.assertEqual(snapshot["stage"], "load")
            self.assertIsNone(snapshot["progress"])
            self.assertFalse(ctx.dry_run)
            return {"batches": 1, "items": batch // 10}

        run = self._pipeline(
            extract=extract, transform=transform, load=load
        ).create_run(
            [1, 2],
            dry_run=False,
            run_root=self.run_root,
            total_batches=2,
        )
        run_holder.append(run)
        self.assertFalse(generator_started)

        summary = await run.execute()

        self.assertEqual(summary.completed_batches, 2)
        self.assertEqual(dict(summary.counts), {"batches": 2, "items": 3})
        with self.assertRaises(TypeError):
            summary.counts["items"] = 99  # type: ignore[index]
        self.assertEqual(
            calls,
            [
                ("extract", 1, {"index": 1, "completed": 0, "total": 2}),
                ("transform", 1, {"index": 1, "completed": 0, "total": 2}),
                ("load", 1, {"index": 1, "completed": 0, "total": 2}),
                ("extract", 2, {"index": 2, "completed": 1, "total": 2}),
                ("transform", 2, {"index": 2, "completed": 1, "total": 2}),
                ("load", 2, {"index": 2, "completed": 1, "total": 2}),
            ],
        )
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "succeeded")
        self.assertFalse(snapshot["dry_run"])
        self.assertEqual(snapshot["stage"], "load")
        self.assertEqual(
            snapshot["batch"], {"index": None, "completed": 2, "total": 2}
        )
        self.assertEqual(
            [event["event"] for event in _read_events(run.events_file)],
            [
                "run_started",
                "batch_completed",
                "batch_completed",
                "run_succeeded",
            ],
        )

    async def test_every_event_records_schema_and_dry_run(self) -> None:
        async def transform(batch: int, *, ctx: PipelineContext) -> int:
            await ctx.progress(1, total=1, message="transformed")
            return batch * 10

        run = self._pipeline(transform=transform).create_run(
            [1], dry_run=True, run_root=self.run_root, total_batches=1
        )

        await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["schema"], "quantmind.etl.batch-run/v1")
        self.assertTrue(snapshot["dry_run"])
        events = _read_events(run.events_file)
        self.assertGreater(len(events), 0)
        for event in events:
            self.assertEqual(event["schema"], "quantmind.etl.batch-event/v1")
            self.assertEqual(event["run_id"], run.id)
            self.assertTrue(event["dry_run"])

    async def test_dry_run_reaches_every_batch_stage_scope(
        self,
    ) -> None:
        calls: list[tuple[str, int | None, bool]] = []

        async def extract(source: list[int], *, ctx: PipelineContext):
            for item in source:
                calls.append(("extract", ctx.batch_index, ctx.dry_run))
                yield item

        async def transform(batch: int, *, ctx: PipelineContext) -> int:
            calls.append(("transform", ctx.batch_index, ctx.dry_run))
            return batch * 10

        async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
            calls.append(("load", ctx.batch_index, ctx.dry_run))
            await ctx.progress(
                1,
                total=1,
                metrics={"planned_records": batch // 10},
            )
            return {"planned_batches": 1, "planned_records": batch // 10}

        run = self._pipeline(
            extract=extract, transform=transform, load=load
        ).create_run(
            [1, 2],
            dry_run=True,
            run_root=self.run_root,
            total_batches=2,
        )

        summary = await run.execute()

        self.assertTrue(run.dry_run)
        self.assertEqual(summary.completed_batches, 2)
        self.assertEqual(
            dict(summary.counts),
            {"planned_batches": 2, "planned_records": 3},
        )
        self.assertEqual(
            calls,
            [
                ("extract", 1, True),
                ("transform", 1, True),
                ("load", 1, True),
                ("extract", 2, True),
                ("transform", 2, True),
                ("load", 2, True),
            ],
        )
        snapshot = _read_json(run.status_file)
        self.assertTrue(snapshot["dry_run"])

    async def test_json_records_escape_surrogate_strings_recursively(
        self,
    ) -> None:
        unsafe = os.fsdecode(b"\xff")
        safe = _json_safe_text(unsafe)

        async def transform(batch: int, *, ctx: PipelineContext) -> int:
            await ctx.progress(
                1,
                total=1,
                message=unsafe,
                metrics={unsafe: unsafe},
            )
            return batch * 10

        async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
            return {unsafe: 1}

        run = self._pipeline(transform=transform, load=load).create_run(
            [1],
            dry_run=False,
            run_root=self.run_root,
            config_summary={unsafe: unsafe},
            total_batches=1,
        )

        summary = await run.execute()

        self.assertEqual(dict(summary.counts), {unsafe: 1})
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "succeeded")
        self.assertEqual(snapshot["config_summary"], {safe: safe})
        progress_events = [
            event
            for event in _read_events(run.events_file)
            if event["event"] == "stage_progress"
        ]
        self.assertEqual(progress_events[-1]["progress"]["message"], safe)
        self.assertEqual(
            progress_events[-1]["progress"]["metrics"], {safe: safe}
        )
        batch_events = [
            event
            for event in _read_events(run.events_file)
            if event["event"] == "batch_completed"
        ]
        self.assertEqual(batch_events[-1]["counts"], {safe: 1})

    async def test_unknown_total_succeeds_with_null_total(self) -> None:
        run = self._pipeline().create_run(
            [1, 2], dry_run=False, run_root=self.run_root
        )

        summary = await run.execute()

        self.assertEqual(summary.completed_batches, 2)
        self.assertEqual(
            _read_json(run.status_file)["batch"],
            {"index": None, "completed": 2, "total": None},
        )

    async def test_known_total_rejects_too_few_batches(self) -> None:
        run = self._pipeline().create_run(
            [1, 2],
            dry_run=False,
            run_root=self.run_root,
            total_batches=3,
        )

        with self.assertRaisesRegex(
            ValueError, "completed batches must equal total_batches"
        ):
            await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["stage"], "extract")
        self.assertEqual(
            snapshot["batch"], {"index": 3, "completed": 2, "total": 3}
        )

    async def test_known_total_rejects_too_many_batches(self) -> None:
        run = self._pipeline().create_run(
            [1, 2],
            dry_run=False,
            run_root=self.run_root,
            total_batches=1,
        )

        with self.assertRaisesRegex(
            ValueError, "extract yielded more batches than total_batches"
        ):
            await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["stage"], "extract")
        self.assertEqual(
            snapshot["batch"], {"index": 2, "completed": 1, "total": 1}
        )

    async def test_progress_resets_per_batch_stage_and_events_include_batch(
        self,
    ) -> None:
        async def extract(source: list[int], *, ctx: PipelineContext):
            for item in source:
                await ctx.progress(1, total=1, message="fetched")
                yield item

        async def transform(batch: int, *, ctx: PipelineContext) -> int:
            await ctx.progress(1, total=1, message="transformed")
            return batch

        async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
            await ctx.progress(1, total=1, message="loaded")
            return {"batches": 1}

        run = self._pipeline(
            extract=extract, transform=transform, load=load
        ).create_run([1, 2], dry_run=False, run_root=self.run_root)

        await run.execute()

        progress_events = [
            event
            for event in _read_events(run.events_file)
            if event["event"] == "stage_progress"
        ]
        self.assertEqual(
            [
                (event["stage"], event["batch_index"])
                for event in progress_events
            ],
            [
                ("extract", 1),
                ("transform", 1),
                ("load", 1),
                ("extract", 2),
                ("transform", 2),
                ("load", 2),
            ],
        )
        self.assertTrue(
            all(
                event["progress"]["completed"] == 1 for event in progress_events
            )
        )

    async def test_current_batch_stage_child_tasks_can_report_progress(
        self,
    ) -> None:
        async def transform(batch: int, *, ctx: PipelineContext) -> int:
            async def first_worker() -> None:
                await ctx.progress(1, total=2)

            async def second_worker() -> None:
                await asyncio.sleep(0)
                await ctx.progress(2, total=2)

            await asyncio.gather(first_worker(), second_worker())
            return batch

        run = self._pipeline(transform=transform).create_run(
            [1], dry_run=False, run_root=self.run_root
        )

        await run.execute()

        progress_events = [
            event
            for event in _read_events(run.events_file)
            if event["event"] == "stage_progress"
        ]
        self.assertEqual(
            [
                (
                    event["stage"],
                    event["batch_index"],
                    event["progress"]["completed"],
                )
                for event in progress_events
            ],
            [("transform", 1, 1), ("transform", 1, 2)],
        )

    async def test_transform_failure_on_batch_two_preserves_completed_one(
        self,
    ) -> None:
        async def transform(batch: int, *, ctx: PipelineContext) -> int:
            if batch == 2:
                raise LookupError("bad transform")
            return batch

        run = self._pipeline(transform=transform).create_run(
            [1, 2], dry_run=False, run_root=self.run_root
        )

        with self.assertRaisesRegex(LookupError, "bad transform"):
            await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["stage"], "transform")
        self.assertEqual(
            snapshot["batch"], {"index": 2, "completed": 1, "total": None}
        )
        self.assertEqual(
            [
                event["event"]
                for event in _read_events(run.events_file)
                if event["event"] == "batch_completed"
            ],
            ["batch_completed"],
        )

    async def test_load_failure_on_batch_two_preserves_completed_one(
        self,
    ) -> None:
        async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
            if batch == 20:
                raise OSError("sink rejected")
            return {"batches": 1}

        run = self._pipeline(load=load).create_run(
            [1, 2], dry_run=False, run_root=self.run_root
        )

        with self.assertRaisesRegex(OSError, "sink rejected"):
            await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["stage"], "load")
        self.assertEqual(
            snapshot["batch"], {"index": 2, "completed": 1, "total": None}
        )

    async def test_cancellation_on_batch_two_preserves_completed_one_without_aclose(
        self,
    ) -> None:
        entered = asyncio.Event()
        closed = False

        async def extract(source: list[int], *, ctx: PipelineContext):
            nonlocal closed
            try:
                for item in source:
                    yield item
            finally:
                closed = True

        async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
            if batch == 20:
                entered.set()
                await asyncio.Event().wait()
            return {"batches": 1}

        run = self._pipeline(extract=extract, load=load).create_run(
            [1, 2, 3], dry_run=False, run_root=self.run_root
        )
        task = asyncio.create_task(run.execute())
        await entered.wait()

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        snapshot = _read_json(run.status_file)
        self.assertFalse(closed)
        self.assertEqual(snapshot["state"], "cancelled")
        self.assertEqual(snapshot["stage"], "load")
        self.assertEqual(
            snapshot["batch"], {"index": 2, "completed": 1, "total": None}
        )

    async def test_cancellation_during_extract_anext_runs_generator_cleanup(
        self,
    ) -> None:
        entered = asyncio.Event()
        closed = False

        async def extract(source: list[int], *, ctx: PipelineContext):
            nonlocal closed
            try:
                entered.set()
                await asyncio.Event().wait()
                yield source[0]
            finally:
                closed = True

        run = self._pipeline(extract=extract).create_run(
            [1], dry_run=False, run_root=self.run_root
        )
        task = asyncio.create_task(run.execute())
        await entered.wait()

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        snapshot = _read_json(run.status_file)
        self.assertTrue(closed)
        self.assertEqual(snapshot["state"], "cancelled")
        self.assertEqual(snapshot["stage"], "extract")
        self.assertEqual(
            snapshot["batch"], {"index": 1, "completed": 0, "total": None}
        )

    async def test_zero_batches_succeeds_without_batch_completed_event(
        self,
    ) -> None:
        run = self._pipeline().create_run(
            [], dry_run=False, run_root=self.run_root, total_batches=0
        )

        summary = await run.execute()

        self.assertEqual(summary.completed_batches, 0)
        self.assertEqual(dict(summary.counts), {})
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "succeeded")
        self.assertIsNone(snapshot["stage"])
        self.assertEqual(
            snapshot["batch"], {"index": None, "completed": 0, "total": 0}
        )
        self.assertEqual(
            [event["event"] for event in _read_events(run.events_file)],
            ["run_started", "run_succeeded"],
        )

    async def test_success_after_final_extract_progress_clears_progress(
        self,
    ) -> None:
        async def extract(source: list[int], *, ctx: PipelineContext):
            yield source[0]
            await ctx.progress(
                1,
                total=1,
                message="checked for another batch",
            )

        run = self._pipeline(extract=extract).create_run(
            [1], dry_run=False, run_root=self.run_root, total_batches=1
        )

        summary = await run.execute()

        self.assertEqual(summary.completed_batches, 1)
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "succeeded")
        self.assertEqual(snapshot["stage"], "load")
        self.assertIsNone(snapshot["progress"])
        self.assertEqual(
            snapshot["batch"], {"index": None, "completed": 1, "total": 1}
        )

    async def test_zero_batch_success_after_extract_progress_clears_stage(
        self,
    ) -> None:
        async def extract(source: list[int], *, ctx: PipelineContext):
            await ctx.progress(1, total=1, message="confirmed empty")
            if False:
                yield source[0]

        run = self._pipeline(extract=extract).create_run(
            [], dry_run=False, run_root=self.run_root, total_batches=0
        )

        summary = await run.execute()

        self.assertEqual(summary.completed_batches, 0)
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "succeeded")
        self.assertIsNone(snapshot["stage"])
        self.assertIsNone(snapshot["progress"])
        self.assertEqual(
            snapshot["batch"], {"index": None, "completed": 0, "total": 0}
        )

    async def test_execute_is_one_shot(self) -> None:
        run = self._pipeline().create_run(
            [1], dry_run=False, run_root=self.run_root
        )
        await run.execute()

        with self.assertRaisesRegex(RuntimeError, "only be called once"):
            await run.execute()

    async def test_extract_context_is_inactive_after_anext_returns(
        self,
    ) -> None:
        captured_contexts: list[PipelineContext] = []

        async def extract(source: list[int], *, ctx: PipelineContext):
            captured_contexts.append(ctx)
            yield source[0]

        async def transform(batch: int, *, ctx: PipelineContext) -> int:
            with self.assertRaisesRegex(RuntimeError, "after the stage ends"):
                await captured_contexts[0].progress(1, total=1)
            return batch

        run = self._pipeline(extract=extract, transform=transform).create_run(
            [1], dry_run=False, run_root=self.run_root
        )

        await run.execute()

        with self.assertRaisesRegex(RuntimeError, "after the stage ends"):
            await captured_contexts[0].progress(1, total=1)

    async def test_stale_background_extract_context_cannot_report_later(
        self,
    ) -> None:
        release = asyncio.Event()
        background_errors: list[str] = []

        async def extract(source: list[int], *, ctx: PipelineContext):
            async def report_later() -> None:
                await release.wait()
                try:
                    await ctx.progress(1, total=1)
                except RuntimeError as exc:
                    background_errors.append(str(exc))

            task = asyncio.create_task(report_later())
            yield source[0]
            release.set()
            await asyncio.sleep(0)
            await ctx.progress(1, total=1)
            yield source[1]
            await task

        run = self._pipeline(extract=extract).create_run(
            [1, 2], dry_run=False, run_root=self.run_root
        )

        await run.execute()

        self.assertEqual(
            background_errors,
            ["progress can only be reported by the active stage scope"],
        )

    async def test_load_count_validation_fails_before_batch_completion(
        self,
    ) -> None:
        async def load(batch: int, *, ctx: PipelineContext):
            return {"items": True}

        run = self._pipeline(load=load).create_run(
            [1], dry_run=False, run_root=self.run_root
        )

        with self.assertRaisesRegex(TypeError, "must be an integer"):
            await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(
            snapshot["batch"], {"index": 1, "completed": 0, "total": None}
        )

    async def test_load_failure_does_not_start_extractor_aclose(
        self,
    ) -> None:
        class ClosingExtractor:
            def __init__(self) -> None:
                self.next_value = 1
                self.closed = False

            def __aiter__(self):
                return self

            async def __anext__(self) -> int:
                if self.next_value > 3:
                    raise StopAsyncIteration
                value = self.next_value
                self.next_value += 1
                return value

            async def aclose(self) -> None:
                self.closed = True
                raise RuntimeError("cleanup failed")

        extractor = ClosingExtractor()

        def extract(source: list[int], *, ctx: PipelineContext):
            return extractor

        async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
            if batch == 20:
                raise LookupError("original load failure")
            return {"batches": 1}

        run = self._pipeline(extract=extract, load=load).create_run(
            [1, 2, 3], dry_run=False, run_root=self.run_root
        )

        with self.assertRaisesRegex(LookupError, "original load failure"):
            await run.execute()

        self.assertFalse(extractor.closed)
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["error"]["type"], "LookupError")

    async def test_hanging_extractor_aclose_is_not_started_on_load_failure(
        self,
    ) -> None:
        class HangingExtractor:
            def __init__(self) -> None:
                self.next_value = 1
                self.close_started = False

            def __aiter__(self):
                return self

            async def __anext__(self) -> int:
                if self.next_value > 3:
                    raise StopAsyncIteration
                value = self.next_value
                self.next_value += 1
                return value

            async def aclose(self) -> None:
                self.close_started = True
                await asyncio.Event().wait()

        extractor = HangingExtractor()

        def extract(source: list[int], *, ctx: PipelineContext):
            return extractor

        async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
            if batch == 20:
                raise LookupError("original load failure")
            return {"batches": 1}

        run = self._pipeline(extract=extract, load=load).create_run(
            [1, 2, 3], dry_run=False, run_root=self.run_root
        )

        with self.assertRaisesRegex(LookupError, "original load failure"):
            await run.execute()

        self.assertFalse(extractor.close_started)
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["error"]["type"], "LookupError")

    async def test_asyncio_run_exits_when_extractor_aclose_swallows_cancel(
        self,
    ) -> None:
        script = r"""
import asyncio
import tempfile
from pathlib import Path

import quantmind.etl._batch as batch_module
from quantmind.etl import BatchETLPipeline, PipelineContext

if hasattr(batch_module, "_EXTRACTOR_CLOSE_TIMEOUT_SECONDS"):
    batch_module._EXTRACTOR_CLOSE_TIMEOUT_SECONDS = 0.01


class StubbornExtractor:
    def __init__(self) -> None:
        self.next_value = 1

    def __aiter__(self):
        return self

    async def __anext__(self) -> int:
        if self.next_value > 2:
            raise StopAsyncIteration
        value = self.next_value
        self.next_value += 1
        return value

    async def aclose(self) -> None:
        while True:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass


async def transform(batch: int, *, ctx: PipelineContext) -> int:
    return batch * 10


async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
    if batch == 20:
        raise LookupError("original load failure")
    return {"batches": 1}


async def main() -> None:
    extractor = StubbornExtractor()

    def extract(source: list[int], *, ctx: PipelineContext):
        return extractor

    with tempfile.TemporaryDirectory() as directory:
        pipeline = BatchETLPipeline(
            "shutdown-regression",
            extract=extract,
            transform=transform,
            load=load,
        )
        run = pipeline.create_run(
            [1, 2],
            dry_run=False,
            run_root=Path(directory),
        )
        try:
            await run.execute()
        except LookupError:
            pass


asyncio.run(main())
"""
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            check=False,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            timeout=3,
        )

        self.assertEqual(
            completed.returncode,
            0,
            f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )

    async def test_hanging_extractor_aclose_is_not_started_on_cancellation(
        self,
    ) -> None:
        entered = asyncio.Event()

        class HangingExtractor:
            def __init__(self) -> None:
                self.next_value = 1
                self.close_started = False

            def __aiter__(self):
                return self

            async def __anext__(self) -> int:
                if self.next_value > 3:
                    raise StopAsyncIteration
                value = self.next_value
                self.next_value += 1
                return value

            async def aclose(self) -> None:
                self.close_started = True
                await asyncio.Event().wait()

        extractor = HangingExtractor()

        def extract(source: list[int], *, ctx: PipelineContext):
            return extractor

        async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
            if batch == 20:
                entered.set()
                await asyncio.Event().wait()
            return {"batches": 1}

        run = self._pipeline(extract=extract, load=load).create_run(
            [1, 2, 3], dry_run=False, run_root=self.run_root
        )
        task = asyncio.create_task(run.execute())
        await entered.wait()

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertFalse(extractor.close_started)
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "cancelled")

    async def test_failed_terminal_event_append_does_not_mask_batch_error(
        self,
    ) -> None:
        original_append = record_module._append_json_line

        def append_event(path: Path, value: dict[str, object]) -> None:
            if value["event"] == "run_failed":
                raise OSError("event sink unavailable")
            original_append(path, value)

        async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
            raise LookupError("batch load failed")

        run = self._pipeline(load=load).create_run(
            [1], dry_run=False, run_root=self.run_root
        )

        with patch(
            "quantmind.etl._record._append_json_line",
            side_effect=append_event,
        ):
            with self.assertRaisesRegex(LookupError, "batch load failed"):
                await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["error"]["type"], "LookupError")

    async def test_run_started_event_append_failure_does_not_leave_running(
        self,
    ) -> None:
        original_append = record_module._append_json_line

        def append_event(path: Path, value: dict[str, object]) -> None:
            if value["event"] == "run_started":
                raise OSError("event sink unavailable")
            original_append(path, value)

        async def load(batch: int, *, ctx: PipelineContext) -> dict[str, int]:
            raise LookupError("batch load failed")

        run = self._pipeline(load=load).create_run(
            [1], dry_run=False, run_root=self.run_root
        )

        with patch(
            "quantmind.etl._record._append_json_line",
            side_effect=append_event,
        ):
            with self.assertRaisesRegex(LookupError, "batch load failed"):
                await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["error"]["type"], "LookupError")
        self.assertNotEqual(snapshot["state"], "running")


class BatchPipelineValidationTests(unittest.TestCase):
    def test_pipeline_name_must_not_be_empty(self) -> None:
        async def extract(value, *, ctx):
            yield value

        async def stage(value, *, ctx):
            return value

        with self.assertRaises(ValueError):
            BatchETLPipeline("", extract=extract, transform=stage, load=stage)

    def test_total_batches_rejects_bool_and_negative_values(self) -> None:
        async def extract(value, *, ctx):
            yield value

        async def stage(value, *, ctx):
            return value

        pipeline = BatchETLPipeline(
            "safe-summary", extract=extract, transform=stage, load=stage
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TypeError, "dry_run"):
                pipeline.create_run("alpha", run_root=Path(directory))
            with self.assertRaisesRegex(TypeError, "dry_run must be"):
                pipeline.create_run(
                    "alpha",
                    dry_run=None,
                    run_root=Path(directory),
                )
            with self.assertRaisesRegex(
                ValueError, r"create_run\(..., dry_run=...\)"
            ):
                pipeline.create_run(
                    "alpha",
                    dry_run=False,
                    run_root=Path(directory),
                    config_summary={"dry_run": False},
                )
            with self.assertRaises(TypeError):
                pipeline.create_run(
                    "alpha",
                    dry_run=False,
                    run_root=Path(directory),
                    total_batches=True,
                )
            with self.assertRaises(ValueError):
                pipeline.create_run(
                    "alpha",
                    dry_run=False,
                    run_root=Path(directory),
                    total_batches=-1,
                )
