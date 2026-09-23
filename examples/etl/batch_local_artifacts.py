"""Run a tiny observable micro-batch ETL that writes local artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from quantmind.etl import BatchETLPipeline, PipelineContext

BATCH_SIZE = 2


async def extract(source: list[str], *, ctx: PipelineContext):
    """Yield small in-memory batches and report each batch before yielding."""
    for offset in range(0, len(source), BATCH_SIZE):
        batch = source[offset : offset + BATCH_SIZE]
        await ctx.progress(
            len(batch),
            total=len(batch),
            message=f"Prepared rows {offset + 1}-{offset + len(batch)}",
        )
        yield batch


async def transform(
    batch: list[str], *, ctx: PipelineContext
) -> list[dict[str, object]]:
    """Build deterministic records for one batch."""
    await ctx.progress(1, total=1, message="Normalized batch")
    return [
        {
            "text": line.strip(),
            "characters": len(line.strip()),
        }
        for line in batch
        if line.strip()
    ]


async def run_example(*, dry_run: bool) -> dict[str, object]:
    """Create the run, print its receipt, then execute all batches."""
    artifact_dir = Path.cwd() / ".quant-mind" / "etl-batch-example"

    async def load(
        records: list[dict[str, object]], *, ctx: PipelineContext
    ) -> dict[str, int]:
        batch_index = ctx.batch_index
        if batch_index is None:
            raise RuntimeError("batch load requires a batch index")
        target = artifact_dir / f"batch-{batch_index:03d}.json"
        content = json.dumps(records, indent=2, sort_keys=True) + "\n"
        byte_count = len(content.encode("utf-8"))
        if ctx.dry_run:
            await ctx.progress(
                1,
                total=1,
                message="Planned batch artifact",
                metrics={
                    "planned_artifacts": 1,
                    "planned_bytes": byte_count,
                    "planned_records": len(records),
                },
            )
            return {
                "planned_artifacts": 1,
                "planned_records": len(records),
            }

        artifact_dir.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        await asyncio.to_thread(temporary.write_text, content, encoding="utf-8")
        temporary.replace(target)
        await ctx.progress(
            1,
            total=1,
            message="Wrote batch artifact",
            metrics={
                "artifacts_written": 1,
                "bytes_written": byte_count,
                "records_written": len(records),
            },
        )
        return {"artifacts_written": 1, "records_written": len(records)}

    source = ["alpha", " beta ", "", "gamma"]
    pipeline = BatchETLPipeline(
        "batch-local-artifacts",
        extract=extract,
        transform=transform,
        load=load,
    )
    run = pipeline.create_run(
        source,
        dry_run=dry_run,
        config_summary={
            "artifact_format": "json",
            "batch_size": BATCH_SIZE,
        },
        total_batches=math.ceil(len(source) / BATCH_SIZE),
    )
    print(run.receipt(), flush=True)
    summary = await run.execute()
    return {
        "dry_run": dry_run,
        "completed_batches": summary.completed_batches,
        "counts": dict(summary.counts),
    }


def _parse_args() -> argparse.Namespace:
    """Parse the runtime run-mode option."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and validate without creating example batch artifacts",
    )
    return parser.parse_args()


async def main() -> None:
    """Run the example in normal or dry-run mode."""
    args = _parse_args()
    summary = await run_example(dry_run=bool(args.dry_run))
    print("summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
