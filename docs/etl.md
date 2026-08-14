# Observable ETL authoring

`quantmind.etl` provides two small async authoring scaffolds with local machine-readable run state:

- `ETLPipeline` performs one whole-run `extract → transform → load` and returns the single load result.
- `BatchETLPipeline` lazily pulls one business batch, transforms it, and loads it before pulling the next batch. It returns an aggregate summary without retaining every batch.

Both require callers to choose `dry_run` when creating a run. Dry-run executes the same stages, including `load`, but authored mutation boundaries must plan or validate without persistent business writes. Concrete domain pipelines belong in the consuming repository; `quantmind.etl` contains only the reusable scaffold. See the [ETL design](../contexts/design/operations/etl.md) for delivery, staging, dry-run, recovery, and selection rules.

## Run one whole-run delivery

Bind three async callables when all processing leads to one final delivery:

```python
from quantmind.etl import ETLPipeline, PipelineContext


async def extract(source: str, *, ctx: PipelineContext) -> list[str]:
    rows = source.splitlines()
    await ctx.progress(len(rows), total=len(rows))
    return rows


async def transform(
    rows: list[str], *, ctx: PipelineContext
) -> list[str]:
    return [row.strip() for row in rows]


async def load(rows: list[str], *, ctx: PipelineContext) -> dict[str, int]:
    if ctx.dry_run:
        return {"planned_rows": len(rows)}
    return {"rows_written": len(rows)}


pipeline = ETLPipeline(
    "line-loader",
    extract=extract,
    transform=transform,
    load=load,
)
dry_run = False
run = pipeline.create_run(
    "one\ntwo",
    dry_run=dry_run,
    config_summary={"drop_empty": False},
)
print(run.receipt(), flush=True)
result = await run.execute()
```

See the runnable [local artifact example](../examples/etl/local_artifact.py) for the smallest complete whole-run path.

## Deliver bounded batches

Use `BatchETLPipeline` when each business-defined batch should be transformed and delivered before the next batch is pulled:

```python
from collections.abc import AsyncIterator, Mapping

from quantmind.etl import BatchETLPipeline, PipelineContext


async def extract(
    rows: list[str], *, ctx: PipelineContext
) -> AsyncIterator[list[str]]:
    batch_size = 2
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        await ctx.progress(len(batch), total=len(batch))
        yield batch


async def transform(
    batch: list[str], *, ctx: PipelineContext
) -> list[str]:
    return [row.strip().upper() for row in batch]


async def load(
    batch: list[str], *, ctx: PipelineContext
) -> Mapping[str, int]:
    if ctx.dry_run:
        return {"planned_records": len(batch)}
    # Persist this batch idempotently, then report count deltas.
    return {"records_written": len(batch)}


pipeline = BatchETLPipeline(
    "batched-line-loader",
    extract=extract,
    transform=transform,
    load=load,
)
dry_run = False
run = pipeline.create_run(
    ["one", "two", "three"],
    dry_run=dry_run,
    total_batches=2,
    config_summary={"batch_size": 2},
)
print(run.receipt(), flush=True)
summary = await run.execute()
```

The framework owns a strictly serial loop. While it awaits the next yielded batch, `run.json.stage` is `extract`; that batch then moves through `transform` and `load` under one one-based `batch.index`. Only a load call that returns successfully increments `batch.completed`. The next batch is not pulled early, so there is no cross-batch pipelining or hidden buffer.

`load` may return `None` or JSON-style non-negative integer count deltas. The runner sums those deltas into `BatchRunSummary.counts` and discards each per-batch result. `total_batches` is optional; when supplied, it is an assertion that fails if extraction yields too many or too few batches.

See [batch local artifacts](../examples/etl/batch_local_artifacts.py) for a network-free, bounded-memory example with idempotent local loads.

## Handle staging inside the owning stage

Decide whether a write is staging or load by downstream consumability, not by whether it touches a database. A staging write persists an intermediate for recovery or reuse while keeping it unavailable to formal downstream consumers; `load` is the boundary after which the intended consumer may treat the result as delivered.

Staging is not a fourth framework stage. Put it inside the `extract` or `transform` callable that creates and owns the intermediate, make the write idempotent, report its real completion through `ctx.progress()`, and suppress the mutation during dry-run. The framework observes the authored stage but does not manage staging artifacts, transactions, rollback, or recovery.

```python
async def extract(source: str, *, ctx: PipelineContext) -> list[str]:
    raw_records = await fetch_raw_records(source)
    if ctx.dry_run:
        await validate_raw_records(raw_records)
        metrics = {"planned_raw_records": len(raw_records)}
        message = "raw staging planned"
    else:
        await raw_store.upsert_many(raw_records)  # Idempotent staging write.
        metrics = {"raw_records_staged": len(raw_records)}
        message = "raw records staged"
    await ctx.progress(
        len(raw_records),
        total=len(raw_records),
        message=message,
        metrics=metrics,
    )
    return raw_records
```

If that write already makes a batch consumable, it is a real `load(batch)` even when the table or object is named `raw`. If an intermediate has an independent downstream consumer, model it as the delivered output of a separate pipeline. See [Distinguish staging from delivery](../contexts/design/operations/etl.md#distinguish-staging-from-delivery) for the canonical decision rule.

## Implement dry-run honestly

`create_run(source, *, dry_run=...)` has no default. Production scripts should read the value from a runtime option such as `--dry-run` and pass that variable, so switching modes never requires editing stage code.

Every stage receives the same read-only `ctx.dry_run` value. Pass it to the repository, gateway, publisher, or storage adapter that owns the mutation decision. The scaffold does not skip `load`; a dry-run load validates the would-be delivery, reports `planned_*` progress or batch counts, and returns a planned path, planned summary, or `None` rather than a reference to a nonexistent artifact. Dry-run forbids persistent business mutation, including staging/checkpoint writes, but still allows reads, parsing, validation, previews, planned counts, and QuantMind's local run-observation files. AI processing may still run and incur cost or rate usage.

## Read local run state

`create_run()` atomically writes `state="created"` before it returns. Snapshots, sparse events, and the receipt include top-level `dry_run`; do not duplicate it inside `config_summary`. The receipt is one JSON line containing the Run ID, `dry_run`, and the absolute `status_file`. By default, each run lives under:

```text
<cwd>/.quant-mind/etl-pipeline-runs/<run-id>/run.json
<cwd>/.quant-mind/etl-pipeline-runs/<run-id>/events.jsonl
```

The four canonical schema IDs are:

- whole-run snapshots: `quantmind.etl.run/v1`;
- whole-run events: `quantmind.etl.event/v1`;
- micro-batch snapshots: `quantmind.etl.batch-run/v1`;
- micro-batch events: `quantmind.etl.batch-event/v1`.

Read `run.json` for the latest snapshot. Check its top-level `dry_run` value before interpreting the result: dry-run `succeeded` means the planned delivery passed validation, not that data was delivered. Its states are `created`, `running`, `succeeded`, `failed`, and `cancelled`. It includes the current macro stage, the executing process's PID, timestamps, safe config summary, latest progress, and a limited error type/message. A batch run additionally includes its current batch index, successfully completed batch count, and optional total. The PID is only a local process-liveness hint; it is not a heartbeat and does not prove forward progress or success.

Call `await ctx.progress(...)` inside meaningful long loops. `completed` means work that really finished and must strictly increase within the current stage, or within the current `(batch, stage)` for batch ETL. Use `total=None` until the total is known. Starting another stage or batch clears the previous progress before user code runs. `ctx.batch_index` is the active one-based index for batch ETL and `None` for whole-run ETL.

Async child tasks created inside the active stage may report progress with the same monotonic counter. Do not leave progress-reporting tasks running after the stage returns: an inherited scope from a completed stage or prior batch is rejected.

Every progress call updates the atomic snapshot. `events.jsonl` keeps only sparse lifecycle events and coalesces progress events emitted within one second; it is not a general logging API. Batch runs emit one `batch_completed` event per batch whose `load` call returned successfully rather than stage-start/stage-complete pairs for every cycle. In a normal run that means the batch was delivered; in a dry-run it means the planned delivery validated.

Only the explicit JSON-scalar `config_summary` allowlist is persisted. Inputs, full configs, intermediate values, individual load results, final results, tracebacks, locals, headers, and response bodies are not serialized automatically.

Every run handle is one-shot. A stage exception records `failed` and is re-raised. Cancellation records `cancelled` on a best-effort basis and then re-raises `asyncio.CancelledError`. Prior successful batch loads are not rolled back. When a batch run fails or is cancelled after a batch has been yielded, the runner does not start the extractor's async `aclose()` cleanup, because a coroutine that ignores cancellation cannot be forcibly bounded on the same event loop and would block `asyncio.run()` shutdown. If cancellation lands while the runner is awaiting the next batch from a normal async generator, Python's own generator cancellation semantics still run that active `finally` cleanup. Concurrency, timeout, retry, extractor cleanup after yielded batches, batch transactionality, and sink idempotency remain explicit responsibilities of the authored pipeline.
