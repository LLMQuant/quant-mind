"""Tests for the runnable ETL dry-run examples."""

import io
import json
import os
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

from examples.etl import batch_local_artifacts, local_artifact


@contextmanager
def _temporary_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


async def _run_without_stdout(coro):
    with redirect_stdout(io.StringIO()):
        return await coro


def _read_run_snapshot(root: Path) -> dict[str, object]:
    snapshots = sorted(
        (root / ".quant-mind" / "etl-pipeline-runs").glob("*/run.json")
    )
    if len(snapshots) != 1:
        raise AssertionError(f"expected one run snapshot, found {snapshots}")
    return json.loads(snapshots[0].read_text(encoding="utf-8"))


class ETLExampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_whole_run_example_dry_run_plans_without_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with _temporary_cwd(root):
                result = await _run_without_stdout(
                    local_artifact.run_example(dry_run=True)
                )

            artifact = root / ".quant-mind" / "etl-example" / "artifact.json"
            self.assertFalse(artifact.exists())
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["planned_artifacts"], 1)
            self.assertIn("planned_artifact", result)

            snapshot = _read_run_snapshot(root)
            self.assertTrue(snapshot["dry_run"])
            self.assertEqual(snapshot["state"], "succeeded")
            self.assertEqual(snapshot["stage"], "load")
            self.assertEqual(
                snapshot["progress"],
                {
                    "completed": 1,
                    "total": 1,
                    "message": "Planned local artifact",
                    "metrics": {
                        "planned_artifacts": 1,
                        "planned_bytes": result["planned_bytes"],
                    },
                },
            )

    async def test_whole_run_example_normal_creates_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with _temporary_cwd(root):
                result = await _run_without_stdout(
                    local_artifact.run_example(dry_run=False)
                )

            artifact = root / ".quant-mind" / "etl-example" / "artifact.json"
            self.assertTrue(artifact.exists())
            self.assertFalse(result["dry_run"])
            self.assertEqual(
                Path(str(result["artifact"])).resolve(), artifact.resolve()
            )
            self.assertEqual(result["artifacts_written"], 1)

            snapshot = _read_run_snapshot(root)
            self.assertFalse(snapshot["dry_run"])
            self.assertEqual(snapshot["state"], "succeeded")

    async def test_batch_example_dry_run_plans_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with _temporary_cwd(root):
                result = await _run_without_stdout(
                    batch_local_artifacts.run_example(dry_run=True)
                )

            artifact_dir = root / ".quant-mind" / "etl-batch-example"
            self.assertFalse(artifact_dir.exists())
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["completed_batches"], 2)
            self.assertEqual(
                result["counts"],
                {
                    "planned_artifacts": 2,
                    "planned_records": 3,
                },
            )

            snapshot = _read_run_snapshot(root)
            self.assertTrue(snapshot["dry_run"])
            self.assertEqual(snapshot["state"], "succeeded")
            self.assertEqual(
                snapshot["batch"],
                {"index": None, "completed": 2, "total": 2},
            )

    async def test_batch_example_normal_creates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with _temporary_cwd(root):
                result = await _run_without_stdout(
                    batch_local_artifacts.run_example(dry_run=False)
                )

            artifact_dir = root / ".quant-mind" / "etl-batch-example"
            self.assertTrue((artifact_dir / "batch-001.json").exists())
            self.assertTrue((artifact_dir / "batch-002.json").exists())
            self.assertFalse(result["dry_run"])
            self.assertEqual(result["completed_batches"], 2)
            self.assertEqual(
                result["counts"],
                {
                    "artifacts_written": 2,
                    "records_written": 3,
                },
            )

            snapshot = _read_run_snapshot(root)
            self.assertFalse(snapshot["dry_run"])
            self.assertEqual(snapshot["state"], "succeeded")
