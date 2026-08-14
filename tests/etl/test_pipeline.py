"""Tests for the observable ETL pipeline scaffold."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import quantmind.etl._record as record_module
from quantmind.etl import ETLPipeline, PipelineContext


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _json_safe_text(value: str) -> str:
    return value.encode("utf-8", errors="backslashreplace").decode("utf-8")


class PipelineRunTests(unittest.IsolatedAsyncioTestCase):
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
    ) -> ETLPipeline[str, str, str, str]:
        async def default_extract(source: str, *, ctx: PipelineContext) -> str:
            return source

        async def default_transform(
            extracted: str, *, ctx: PipelineContext
        ) -> str:
            return extracted.upper()

        async def default_load(
            transformed: str, *, ctx: PipelineContext
        ) -> str:
            return f"loaded:{transformed}"

        return ETLPipeline(
            "test-pipeline",
            extract=extract or default_extract,
            transform=transform or default_transform,
            load=load or default_load,
        )

    async def test_create_run_writes_created_snapshot_before_return(
        self,
    ) -> None:
        source = "private-source"
        run = self._pipeline().create_run(
            source,
            dry_run=True,
            run_root=self.run_root,
            config_summary={"window_days": 30},
        )

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["schema"], "quantmind.etl.run/v1")
        self.assertTrue(snapshot["dry_run"])
        self.assertTrue(run.status_file.is_absolute())
        self.assertEqual(snapshot["state"], "created")
        self.assertIsNone(snapshot["stage"])
        self.assertIsNone(snapshot["pid"])
        self.assertEqual(snapshot["config_summary"], {"window_days": 30})
        self.assertNotIn(source, run.status_file.read_text(encoding="utf-8"))
        self.assertEqual(run.events_file.read_text(encoding="utf-8"), "")

    async def test_success_runs_fixed_stages_and_records_lifecycle(
        self,
    ) -> None:
        calls: list[str] = []

        async def extract(source: str, *, ctx: PipelineContext) -> str:
            calls.append(ctx.stage)
            self.assertIsNone(ctx.batch_index)
            self.assertFalse(ctx.dry_run)
            await ctx.progress(1, total=1, message="read")
            return source

        async def transform(value: str, *, ctx: PipelineContext) -> str:
            calls.append(ctx.stage)
            self.assertIsNone(ctx.batch_index)
            self.assertFalse(ctx.dry_run)
            return value.upper()

        async def load(value: str, *, ctx: PipelineContext) -> str:
            calls.append(ctx.stage)
            self.assertIsNone(ctx.batch_index)
            self.assertFalse(ctx.dry_run)
            return f"loaded:{value}"

        run = self._pipeline(
            extract=extract,
            transform=transform,
            load=load,
        ).create_run("alpha", dry_run=False, run_root=self.run_root)

        result = await run.execute()

        self.assertEqual(result, "loaded:ALPHA")
        self.assertEqual(calls, ["extract", "transform", "load"])
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "succeeded")
        self.assertFalse(snapshot["dry_run"])
        self.assertEqual(snapshot["stage"], "load")
        self.assertEqual(snapshot["pid"], os.getpid())
        self.assertIsNone(snapshot["error"])
        self.assertNotIn(result, run.status_file.read_text(encoding="utf-8"))
        self.assertEqual(
            [event["event"] for event in _read_events(run.events_file)],
            [
                "run_started",
                "stage_started",
                "stage_progress",
                "stage_completed",
                "stage_started",
                "stage_completed",
                "stage_started",
                "stage_completed",
                "run_succeeded",
            ],
        )

    async def test_every_event_records_schema_and_dry_run(self) -> None:
        async def extract(source: str, *, ctx: PipelineContext) -> str:
            await ctx.progress(1, total=1, message="read")
            return source

        run = self._pipeline(extract=extract).create_run(
            "alpha", dry_run=True, run_root=self.run_root
        )

        await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["schema"], "quantmind.etl.run/v1")
        self.assertTrue(snapshot["dry_run"])
        events = _read_events(run.events_file)
        self.assertGreater(len(events), 0)
        for event in events:
            self.assertEqual(event["schema"], "quantmind.etl.event/v1")
            self.assertEqual(event["run_id"], run.id)
            self.assertTrue(event["dry_run"])

    async def test_dry_run_reaches_every_stage_and_still_calls_load(
        self,
    ) -> None:
        calls: list[tuple[str, bool]] = []

        async def extract(source: str, *, ctx: PipelineContext) -> str:
            calls.append((ctx.stage, ctx.dry_run))
            return source

        async def transform(value: str, *, ctx: PipelineContext) -> str:
            calls.append((ctx.stage, ctx.dry_run))
            return value.upper()

        async def load(value: str, *, ctx: PipelineContext) -> str:
            calls.append((ctx.stage, ctx.dry_run))
            await ctx.progress(
                1,
                total=1,
                metrics={"planned_records": 1},
            )
            return f"planned:{value}"

        run = self._pipeline(
            extract=extract,
            transform=transform,
            load=load,
        ).create_run("alpha", dry_run=True, run_root=self.run_root)

        result = await run.execute()

        self.assertEqual(result, "planned:ALPHA")
        self.assertEqual(
            calls,
            [("extract", True), ("transform", True), ("load", True)],
        )
        snapshot = _read_json(run.status_file)
        self.assertTrue(snapshot["dry_run"])
        self.assertEqual(
            snapshot["progress"],
            {
                "completed": 1,
                "total": 1,
                "message": None,
                "metrics": {"planned_records": 1},
            },
        )

    async def test_json_records_escape_surrogate_strings_recursively(
        self,
    ) -> None:
        unsafe = os.fsdecode(b"\xff")
        safe = _json_safe_text(unsafe)

        async def load(value: str, *, ctx: PipelineContext) -> str:
            await ctx.progress(
                1,
                total=1,
                message=unsafe,
                metrics={unsafe: unsafe},
            )
            return value

        run = self._pipeline(load=load).create_run(
            "alpha",
            dry_run=False,
            run_root=self.run_root,
            config_summary={unsafe: unsafe},
        )

        await run.execute()

        snapshot_text = run.status_file.read_text(encoding="utf-8")
        snapshot = json.loads(snapshot_text)
        self.assertEqual(snapshot["state"], "succeeded")
        self.assertEqual(snapshot["config_summary"], {safe: safe})
        self.assertEqual(snapshot["progress"]["message"], safe)
        self.assertEqual(snapshot["progress"]["metrics"], {safe: safe})
        progress_events = [
            event
            for event in _read_events(run.events_file)
            if event["event"] == "stage_progress"
        ]
        self.assertEqual(progress_events[-1]["progress"]["message"], safe)
        self.assertEqual(
            progress_events[-1]["progress"]["metrics"], {safe: safe}
        )

    async def test_failure_after_surrogate_progress_records_failed_json(
        self,
    ) -> None:
        unsafe = os.fsdecode(b"\xff")
        safe = _json_safe_text(unsafe)

        async def transform(value: str, *, ctx: PipelineContext) -> str:
            await ctx.progress(
                1,
                total=1,
                message=unsafe,
                metrics={unsafe: unsafe},
            )
            raise LookupError("stage failed")

        run = self._pipeline(transform=transform).create_run(
            "alpha",
            dry_run=False,
            run_root=self.run_root,
            config_summary={unsafe: unsafe},
        )

        with self.assertRaisesRegex(LookupError, "stage failed"):
            await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["config_summary"], {safe: safe})
        self.assertEqual(snapshot["progress"]["message"], safe)
        self.assertEqual(snapshot["progress"]["metrics"], {safe: safe})
        self.assertEqual(snapshot["error"]["type"], "LookupError")
        self.assertEqual(
            _read_events(run.events_file)[-1]["event"], "run_failed"
        )

    async def test_normal_and_dry_runs_are_independent(self) -> None:
        async def load(value: str, *, ctx: PipelineContext) -> str:
            if ctx.dry_run:
                return f"planned:{value}"
            return f"loaded:{value}"

        pipeline = self._pipeline(load=load)
        normal = pipeline.create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )
        dry = pipeline.create_run("beta", dry_run=True, run_root=self.run_root)

        normal_result = await normal.execute()
        dry_result = await dry.execute()

        self.assertNotEqual(normal.id, dry.id)
        self.assertFalse(normal.dry_run)
        self.assertTrue(dry.dry_run)
        self.assertEqual(normal_result, "loaded:ALPHA")
        self.assertEqual(dry_result, "planned:BETA")
        normal_snapshot = _read_json(normal.status_file)
        dry_snapshot = _read_json(dry.status_file)
        self.assertEqual(normal_snapshot["state"], "succeeded")
        self.assertEqual(dry_snapshot["state"], "succeeded")
        self.assertFalse(normal_snapshot["dry_run"])
        self.assertTrue(dry_snapshot["dry_run"])

    async def test_failure_records_limited_error_and_reraises(self) -> None:
        async def fail(value: str, *, ctx: PipelineContext) -> str:
            raise LookupError("missing item")

        run = self._pipeline(transform=fail).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        with self.assertRaisesRegex(LookupError, "missing item"):
            await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["stage"], "transform")
        self.assertEqual(
            snapshot["error"],
            {"message": "missing item", "type": "LookupError"},
        )
        self.assertEqual(
            _read_events(run.events_file)[-1]["event"], "run_failed"
        )

    async def test_surrogate_error_message_records_json_safely(
        self,
    ) -> None:
        class SurrogateMessageError(Exception):
            def __str__(self) -> str:
                return "bad value \ud800"

        original_error = SurrogateMessageError()

        async def fail(value: str, *, ctx: PipelineContext) -> str:
            raise original_error

        run = self._pipeline(transform=fail).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        with self.assertRaises(SurrogateMessageError) as raised:
            await run.execute()

        self.assertIs(raised.exception, original_error)
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(
            snapshot["error"],
            {"message": "bad value \\ud800", "type": "SurrogateMessageError"},
        )
        self.assertEqual(
            _read_events(run.events_file)[-1]["error"],
            {"message": "bad value \\ud800", "type": "SurrogateMessageError"},
        )

    async def test_failed_terminal_event_append_does_not_mask_stage_error(
        self,
    ) -> None:
        original_append = record_module._append_json_line

        def append_event(path: Path, value: dict[str, object]) -> None:
            if value["event"] == "run_failed":
                raise OSError("event sink unavailable")
            original_append(path, value)

        async def fail(value: str, *, ctx: PipelineContext) -> str:
            raise LookupError("stage failed")

        run = self._pipeline(transform=fail).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        with patch(
            "quantmind.etl._record._append_json_line",
            side_effect=append_event,
        ):
            with self.assertRaisesRegex(LookupError, "stage failed"):
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

        async def fail(value: str, *, ctx: PipelineContext) -> str:
            raise LookupError("stage failed")

        run = self._pipeline(transform=fail).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        with patch(
            "quantmind.etl._record._append_json_line",
            side_effect=append_event,
        ):
            with self.assertRaisesRegex(LookupError, "stage failed"):
                await run.execute()

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["error"]["type"], "LookupError")
        self.assertNotEqual(snapshot["state"], "running")

    async def test_failed_terminal_snapshot_write_does_not_mask_stage_error(
        self,
    ) -> None:
        original_write = record_module._write_json_atomic
        original_error = LookupError("stage failed")

        def write_snapshot(path: Path, value: dict[str, object]) -> None:
            if value["state"] == "failed":
                raise OSError("snapshot unavailable")
            original_write(path, value)

        async def fail(value: str, *, ctx: PipelineContext) -> str:
            raise original_error

        run = self._pipeline(transform=fail).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        with patch(
            "quantmind.etl._record._write_json_atomic",
            side_effect=write_snapshot,
        ):
            with self.assertRaises(LookupError) as raised:
                await run.execute()

        self.assertIs(raised.exception, original_error)
        snapshot = _read_json(run.status_file)
        self.assertNotEqual(snapshot["state"], "succeeded")

    async def test_failed_terminal_snapshot_survives_pending_progress_event_error(
        self,
    ) -> None:
        original_append = record_module._append_json_line
        original_error = LookupError("stage failed")

        def append_event(path: Path, value: dict[str, object]) -> None:
            if (
                value["event"] == "stage_progress"
                and value["progress"]["completed"] == 2
            ):
                raise OSError("progress journal unavailable")
            original_append(path, value)

        async def fail_after_pending_progress(
            value: str, *, ctx: PipelineContext
        ) -> str:
            await ctx.progress(1, total=3)
            await ctx.progress(2, total=3)
            raise original_error

        run = self._pipeline(extract=fail_after_pending_progress).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        with (
            patch(
                "quantmind.etl._record._append_json_line",
                side_effect=append_event,
            ),
            patch(
                "quantmind.etl._record._monotonic_seconds",
                side_effect=[10.0, 10.1],
            ),
        ):
            with self.assertRaises(LookupError) as raised:
                await run.execute()

        self.assertIs(raised.exception, original_error)
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["error"]["type"], "LookupError")

    async def test_cancelled_terminal_snapshot_write_preserves_cancelled_error(
        self,
    ) -> None:
        original_write = record_module._write_json_atomic
        entered = asyncio.Event()

        def write_snapshot(path: Path, value: dict[str, object]) -> None:
            if value["state"] == "cancelled":
                raise OSError("snapshot unavailable")
            original_write(path, value)

        async def wait_forever(value: str, *, ctx: PipelineContext) -> str:
            entered.set()
            await asyncio.Event().wait()
            return value

        run = self._pipeline(extract=wait_forever).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )
        task = asyncio.create_task(run.execute())
        await entered.wait()

        with patch(
            "quantmind.etl._record._write_json_atomic",
            side_effect=write_snapshot,
        ):
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        snapshot = _read_json(run.status_file)
        self.assertNotEqual(snapshot["state"], "succeeded")

    async def test_succeeded_snapshot_makes_terminal_event_best_effort(
        self,
    ) -> None:
        original_append = record_module._append_json_line

        def append_event(path: Path, value: dict[str, object]) -> None:
            if value["event"] == "run_succeeded":
                raise OSError("event sink unavailable")
            original_append(path, value)

        run = self._pipeline().create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        with patch(
            "quantmind.etl._record._append_json_line",
            side_effect=append_event,
        ):
            result = await run.execute()

        self.assertEqual(result, "loaded:ALPHA")
        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "succeeded")
        self.assertEqual(
            [event["event"] for event in _read_events(run.events_file)][-1],
            "stage_completed",
        )

    async def test_cancellation_records_terminal_state_then_reraises(
        self,
    ) -> None:
        entered = asyncio.Event()

        async def wait_forever(value: str, *, ctx: PipelineContext) -> str:
            entered.set()
            await asyncio.Event().wait()
            return value

        run = self._pipeline(extract=wait_forever).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )
        task = asyncio.create_task(run.execute())
        await entered.wait()

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        snapshot = _read_json(run.status_file)
        self.assertEqual(snapshot["state"], "cancelled")
        self.assertEqual(snapshot["stage"], "extract")
        self.assertIsNone(snapshot["error"])
        self.assertEqual(
            _read_events(run.events_file)[-1]["event"], "run_cancelled"
        )

    async def test_receipt_is_one_line_json_with_absolute_status_file(
        self,
    ) -> None:
        run = self._pipeline().create_run(
            "alpha", dry_run=True, run_root=self.run_root
        )

        receipt = run.receipt()
        parsed = json.loads(receipt)
        self.assertNotIn("\n", receipt)
        self.assertEqual(parsed["event"], "etl_run_created")
        self.assertEqual(parsed["run_id"], run.id)
        self.assertTrue(parsed["dry_run"])
        self.assertEqual(parsed["status_file"], str(run.status_file))
        self.assertTrue(Path(parsed["status_file"]).is_absolute())

    async def test_new_stage_clears_previous_stage_progress(self) -> None:
        run_holder = []
        observed: list[tuple[str, object]] = []

        async def extract(source: str, *, ctx: PipelineContext) -> str:
            await ctx.progress(2, total=2)
            return source

        async def transform(value: str, *, ctx: PipelineContext) -> str:
            snapshot = _read_json(run_holder[0].status_file)
            observed.append((str(snapshot["stage"]), snapshot["progress"]))
            await ctx.progress(1, total=None)
            return value

        async def load(value: str, *, ctx: PipelineContext) -> str:
            snapshot = _read_json(run_holder[0].status_file)
            observed.append((str(snapshot["stage"]), snapshot["progress"]))
            return value

        run = self._pipeline(
            extract=extract,
            transform=transform,
            load=load,
        ).create_run("alpha", dry_run=False, run_root=self.run_root)
        run_holder.append(run)

        await run.execute()

        self.assertEqual(observed, [("transform", None), ("load", None)])

    async def test_progress_requires_strict_real_completion(self) -> None:
        async def invalid(source: str, *, ctx: PipelineContext) -> str:
            await ctx.progress(1, total=3)
            await ctx.progress(1, total=3)
            return source

        run = self._pipeline(extract=invalid).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        with self.assertRaisesRegex(ValueError, "strictly increase"):
            await run.execute()
        self.assertEqual(_read_json(run.status_file)["state"], "failed")

    async def test_progress_rejects_completed_above_total(self) -> None:
        async def invalid(source: str, *, ctx: PipelineContext) -> str:
            await ctx.progress(2, total=1)
            return source

        run = self._pipeline(extract=invalid).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        with self.assertRaisesRegex(ValueError, "total must be >= completed"):
            await run.execute()

    async def test_progress_total_none_means_unknown_until_known(self) -> None:
        async def invalid(source: str, *, ctx: PipelineContext) -> str:
            await ctx.progress(1, total=None)
            await ctx.progress(2, total=3)
            await ctx.progress(3, total=None)
            return source

        run = self._pipeline(extract=invalid).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        with self.assertRaisesRegex(ValueError, "cannot become unknown"):
            await run.execute()

    async def test_stage_child_tasks_can_report_progress(self) -> None:
        async def extract(source: str, *, ctx: PipelineContext) -> str:
            async def first_worker() -> None:
                await ctx.progress(1, total=2)

            async def second_worker() -> None:
                await asyncio.sleep(0)
                await ctx.progress(2, total=2)

            await asyncio.gather(first_worker(), second_worker())
            return source

        run = self._pipeline(extract=extract).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        await run.execute()

        progress_events = [
            event
            for event in _read_events(run.events_file)
            if event["event"] == "stage_progress"
        ]
        self.assertEqual(
            [event["progress"]["completed"] for event in progress_events],
            [1, 2],
        )

    async def test_progress_events_are_coalesced_but_snapshot_is_current(
        self,
    ) -> None:
        async def extract(source: str, *, ctx: PipelineContext) -> str:
            await ctx.progress(1, total=3)
            await ctx.progress(2, total=3)
            await ctx.progress(3, total=3)
            return source

        run = self._pipeline(extract=extract).create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )

        with patch(
            "quantmind.etl._record._monotonic_seconds",
            side_effect=[10.0, 10.1, 10.2],
        ):
            await run.execute()

        progress_events = [
            event
            for event in _read_events(run.events_file)
            if event["event"] == "stage_progress"
        ]
        self.assertEqual(len(progress_events), 2)
        self.assertEqual(
            [event["progress"]["completed"] for event in progress_events],
            [1, 3],
        )

    async def test_execute_is_one_shot(self) -> None:
        run = self._pipeline().create_run(
            "alpha", dry_run=False, run_root=self.run_root
        )
        await run.execute()

        with self.assertRaisesRegex(RuntimeError, "only be called once"):
            await run.execute()


class PipelineValidationTests(unittest.TestCase):
    def test_pipeline_name_must_not_be_empty(self) -> None:
        async def stage(value, *, ctx):
            return value

        with self.assertRaises(ValueError):
            ETLPipeline("", extract=stage, transform=stage, load=stage)

    def test_config_summary_is_a_json_scalar_allowlist(self) -> None:
        async def stage(value, *, ctx):
            return value

        pipeline = ETLPipeline(
            "safe-summary", extract=stage, transform=stage, load=stage
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
                    config_summary={"dry_run": True},
                )
            with self.assertRaises(TypeError):
                pipeline.create_run(
                    "alpha",
                    dry_run=False,
                    run_root=Path(directory),
                    config_summary={"secret": {"nested": "value"}},
                )
            with self.assertRaises(ValueError):
                pipeline.create_run(
                    "alpha",
                    dry_run=False,
                    run_root=Path(directory),
                    config_summary={"ratio": float("nan")},
                )
