"""Run a tiny observable ETL that writes one idempotent local artifact."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from quantmind.etl import ETLPipeline, PipelineContext

LoadResult = dict[str, object]


async def extract(source: Path, *, ctx: PipelineContext) -> str:
    """Read one local text source."""
    text = await asyncio.to_thread(source.read_text, encoding="utf-8")
    await ctx.progress(1, total=1, message="Read local source")
    return text


async def transform(text: str, *, ctx: PipelineContext) -> dict[str, object]:
    """Build a deterministic summary."""
    lines = [line for line in text.splitlines() if line.strip()]
    return {"non_empty_lines": len(lines), "characters": len(text)}


async def run_example(*, dry_run: bool) -> LoadResult:
    """Create the run, print its receipt, then execute all stages."""
    artifact = Path.cwd() / ".quant-mind" / "etl-example" / "artifact.json"

    async def load(
        summary: dict[str, object], *, ctx: PipelineContext
    ) -> LoadResult:
        content = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        byte_count = len(content.encode("utf-8"))
        if ctx.dry_run:
            await ctx.progress(
                1,
                total=1,
                message="Planned local artifact",
                metrics={
                    "planned_artifacts": 1,
                    "planned_bytes": byte_count,
                },
            )
            return {
                "dry_run": True,
                "planned_artifact": str(artifact),
                "planned_artifacts": 1,
                "planned_bytes": byte_count,
            }

        artifact.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(artifact.write_text, content, encoding="utf-8")
        await ctx.progress(
            1,
            total=1,
            message="Wrote local artifact",
            metrics={
                "artifacts_written": 1,
                "bytes_written": byte_count,
            },
        )
        return {
            "dry_run": False,
            "artifact": str(artifact),
            "artifacts_written": 1,
            "bytes_written": byte_count,
        }

    pipeline = ETLPipeline(
        "local-source-summary",
        extract=extract,
        transform=transform,
        load=load,
    )
    run = pipeline.create_run(
        Path(__file__),
        dry_run=dry_run,
        config_summary={"artifact_format": "json"},
    )
    print(run.receipt(), flush=True)
    return await run.execute()


def _parse_args() -> argparse.Namespace:
    """Parse the runtime run-mode option."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and validate without creating the example artifact",
    )
    return parser.parse_args()


async def main() -> None:
    """Run the example in normal or dry-run mode."""
    args = _parse_args()
    result = await run_example(dry_run=bool(args.dry_run))
    print("result=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
