# Author observable whole-run and micro-batch ETL

## Quick Summary

- **Purpose**: Define the two explicit execution shapes in `quantmind.etl`, their delivery semantics, and the local observation contract shared by both.
- **Read when**: Choosing between `ETLPipeline` and `BatchETLPipeline`, changing their execution or run-record behavior, or adopting either scaffold in a consuming repository.
- **Owner**: `quantmind.etl`; concrete domain pipelines remain in consuming repositories or focused examples.
- **Status**: Current. `ETLPipeline` and `BatchETLPipeline` are implemented together in the active ETL change.

## Contents

- [Keep two execution shapes explicit](#keep-two-execution-shapes-explicit)
- [Run one delivery through ETLPipeline](#run-one-delivery-through-etlpipeline)
- [Deliver bounded batches through BatchETLPipeline](#deliver-bounded-batches-through-batchetlpipeline)
- [Make dry-run an explicit run property](#make-dry-run-an-explicit-run-property)
- [Observe lifecycle and real progress locally](#observe-lifecycle-and-real-progress-locally)
- [Distinguish staging from delivery](#distinguish-staging-from-delivery)
- [Choose a scaffold by delivery cardinality](#choose-a-scaffold-by-delivery-cardinality)
- [Keep execution policy in the authored pipeline](#keep-execution-policy-in-the-authored-pipeline)
- [Defer concurrent and durable orchestration](#defer-concurrent-and-durable-orchestration)

## Keep two execution shapes explicit

`quantmind.etl` is a stdlib-only authoring and observation scaffold, not a workflow engine. It exposes two parallel composition-based classes because delivery cardinality changes execution and observation semantics:

- `ETLPipeline` performs one whole-run `extract → transform → load` and returns that single load result.
- `BatchETLPipeline` repeatedly pulls one business-defined batch, transforms it, and loads it before pulling the next batch. It returns a bounded summary rather than retaining every load result.

Neither class inherits the other. A callable returning an async iterable never makes `ETLPipeline` switch modes implicitly: return-value sniffing would misclassify legitimate values and make the run-record contract conditional. Both classes reuse a package-private local record layer because atomic snapshots, sparse JSONL events, and progress coalescing now have two real callers.

Existing `quantmind.flows` remain pure `input → self-contained artifact` operations and neither inherit nor depend on these ETL scaffolds. Concrete ingestion pipelines belong in their consuming repository; `quantmind.etl` contains only reusable execution and observation behavior.

## Run one delivery through ETLPipeline

`ETLPipeline` binds three async callables and invokes each exactly once:

```text
source → extract → extracted value → transform → final value → load → result
```

It is the smallest shape when the operation has one delivery boundary. Each run is one-shot. The pipeline instance stores only its stable name and callables; input, intermediate values, result, stage, and progress remain run-local and are never persisted automatically.

## Deliver bounded batches through BatchETLPipeline

`BatchETLPipeline` binds an async batch producer plus async transform and load callables. The framework owns a strictly serial loop:

```text
pull batch 1 as extract → transform batch 1 → load batch 1
pull batch 2 as extract → transform batch 2 → load batch 2
...
```

The authored extractor decides batch boundaries and yields them lazily. The framework does not automatically slice inputs. It sets `stage="extract"` while awaiting the next yielded batch, then switches to `transform` and `load` for that same one-based batch index. It never waits for every transformed batch in memory, and v-next does not overlap extraction or transformation of batch N+1 with loading batch N.

A batch becomes completed only after its load callable returns successfully. A failed or cancelled load may already have produced business side effects; the scaffold cannot prove transactionality. The load implementation must make one batch an idempotent or atomic delivery unit so rerunning the whole run can safely skip or repeat it. The framework does not checkpoint a cursor or resume inside a prior run.

Each successful load returns optional non-negative count deltas. The runner adds those deltas into a final `BatchRunSummary` and discards the per-batch return value, preserving bounded memory. The summary reports only facts the framework observed: successfully completed batch count and accumulated declared counts.

An optional known batch total is an assertion, not an estimate. Yielding more batches than declared, or exhausting the producer before the declared total completes, fails the run. Unknown totals remain `null`.

## Make dry-run an explicit run property

Both pipeline shapes require callers to choose `dry_run` when they create a run:

```python
run = pipeline.create_run(source, dry_run=dry_run)
```

There is deliberately no `False` default. A production script or CLI must obtain the value from a runtime option and pass that variable so switching modes never requires editing pipeline code; tests, notebooks, and fixed-purpose one-off scripts may pass a Boolean literal. The run-level value is immutable, shared by every stage and batch, and available as the read-only `PipelineContext.dry_run` property. Stage authors pass it through to the repository, gateway, publisher, or other capability that owns the actual mutation decision instead of scattering temporary switches through orchestration code.

Dry-run executes the complete stage shape. The scaffold never skips `load`: a dry-run load still validates the would-be delivery, computes planned counts, reports progress, and may expose errors visible only at the delivery boundary. With `ctx.dry_run=True`, however, the authored pipeline must prevent every persistent business mutation, including database changes, storage or artifact writes, staging and checkpoint writes, publishing, queueing, webhooks, and final delivery. Reads, fetching, parsing, normalization, pruning, AI processing, previews, planned counts, validation, and QuantMind's own local run-observation files remain allowed. AI processing may still incur cost and rate usage; dry-run is not a zero-cost mode.

One load signature serves both modes, so its output type must honestly represent both. In dry-run it returns a planned path, planned summary, or `None`, never a reference that implies a nonexistent artifact was delivered. Dry-run progress metrics and batch count deltas use a `planned_*` prefix or an equally explicit planned name; names that assert real delivery, such as `rows_written`, are reserved for normal runs.

`dry_run` is a formal top-level run field, not configuration metadata. It appears in `run.json` from the initial `created` snapshot onward and in the creation receipt, while the schema IDs remain at v1 because readers already tolerate additive fields. `config_summary["dry_run"]` is rejected with guidance to use `create_run(..., dry_run=...)`, preventing two competing sources of truth. An observer reads the top-level flag before interpreting the terminal state: a dry-run `succeeded` means the plan completed validation, not that data was delivered.

This flag establishes a narrow admission rule for future execution-semantic parameters. A run-level flag enters `create_run()` only when it both changes external side-effect semantics and is necessary for an observer to interpret the run correctly. `dry_run` satisfies both conditions; tuning such as `verbose`, `fast_mode`, or `force` remains authored-pipeline configuration rather than expanding the framework signature.

## Observe lifecycle and real progress locally

`create_run()` allocates a Run ID and atomically writes `state="created"` before returning. `receipt()` provides that Run ID and the absolute `run.json` path as one JSON line, which a wrapper prints and flushes before `execute()`.

Both shapes keep an atomic latest snapshot in `run.json` and a sparse lifecycle journal in `events.jsonl`. They use distinct schema IDs so an observer never has to infer execution shape from values:

- whole-run snapshots: `quantmind.etl.run/v1`;
- whole-run events: `quantmind.etl.event/v1`;
- micro-batch snapshots: `quantmind.etl.batch-run/v1`, with `batch.index`, successfully completed load count `batch.completed`, and optional `batch.total`;
- micro-batch events: `quantmind.etl.batch-event/v1`.

`run.json` is the authoritative state: after a terminal snapshot commits, the terminal JSONL append is best-effort and cannot change the `execute()` outcome, while failure and cancellation observation errors preserve the original exception.

In a normal micro-batch run, completed loads represent delivery; in dry-run they represent validated planned deliveries.

Micro-batch events record run lifecycle, explicitly reported stage progress, and one `batch_completed` event per meaningful delivery unit or validated planned delivery in dry-run. They do not emit stage-start/stage-complete pairs for every batch. Batch size is expected to represent tens or hundreds of meaningful delivery units; producing thousands of tiny batches is an authored-pipeline configuration problem rather than a reason to turn the sparse journal into arbitrary logging.

`PipelineContext.progress()` always means real finished work. Its strict monotonicity and known-total rules apply within one active stage for `ETLPipeline`, and within one active `(batch index, stage)` for `BatchETLPipeline`. Every stage or batch switch clears the previous progress snapshot before user code runs. In batch mode, `ctx.batch_index` identifies the current one-based batch; it is `None` for whole-run ETL.

Child tasks created inside the active stage inherit its progress scope and may report real completion, subject to the same monotonic counter. A task that outlives that stage or batch keeps the old scope and its later progress call is rejected, so a stale producer cannot write into the next batch's snapshot.

Inputs, complete configuration, intermediate values, individual load results, tracebacks, headers, response bodies, and arbitrary logs are not written. Only the caller's explicit JSON-scalar `config_summary`, framework lifecycle fields, limited error type/message, and explicit progress/count measurements appear.

## Distinguish staging from delivery

Staging is intentionally not a fourth framework stage, and the scaffold exposes no `stage()` or `staging()` API. The authored `extract` or `transform` operation that creates and owns an intermediate also owns any staging write needed to recover or reuse it.

`load` marks the delivery boundary. In a normal run, after it succeeds the pipeline's intended downstream consumer may treat that output as delivered; in dry-run it validates or plans the same boundary without delivering. A database call is not automatically a load. Extract or transform may perform staging writes when they persist an intermediate for recovery or reuse while keeping it unavailable to formal downstream consumers.

Staging writes are permitted inside the owning extract or transform stage of a normal run when they are idempotent and their real completion is visible through progress. Dry-run must plan or validate them without writing. They remain business behavior: the scaffold provides no transaction, rollback, exactly-once, or artifact-management guarantee.

If a batch write makes the final product consumable, it is a real batch load even when the target table is named `raw`. Repeated `transform(batch) → load(batch)` delivery belongs in `BatchETLPipeline`; hiding those loads inside a whole-run transform would make `run.json.stage` dishonest.

## Choose a scaffold by delivery cardinality

Use this selection test:

| Delivery shape | Scaffold |
|---|---|
| One final delivery after all processing | `ETLPipeline` |
| Repeated bounded deliveries, one per business batch | `BatchETLPipeline` |
| Intermediate durable writes that are not consumable products | Keep them as idempotent staging inside the owning stage |
| An intermediate is itself an independently consumed product | Split into two pipelines and pass run A's artifact to run B |
| Named steps need independent retry/resume or form a graph | Neither current scaffold; evaluate a later ETL DAG only after that requirement is concrete |

A write followed by another decision does not alone imply a workflow. Ask whether the write is staging or delivery, and how many delivery units the run intentionally publishes.

## Keep execution policy in the authored pipeline

The scaffold records what runs; it does not select concurrency, timeout, retry, transaction, or idempotency policy. Stage-local timeouts fail the stage when surfaced as `TimeoutError`; cancelling `execute()` from an outer deadline records `cancelled` on a best-effort basis and re-raises `CancelledError`. Retry remains explicit and must respect side-effect safety.

Batch extractor cleanup is intentionally not a hidden bounded policy. If failure or cancellation happens after a batch has already been yielded, the runner does not proactively start the extractor's async `aclose()` cleanup: once such an awaitable is running on the current event loop, asyncio cannot force it to stop if it swallows cancellation, and leaving it detached can block `asyncio.run()` shutdown. If cancellation lands while awaiting the next batch from a normal async generator, Python's own generator cancellation semantics still run cleanup for that active await. Cleanup after yielded batches remains an authored-pipeline responsibility.

Cancellation stops at the active await. A batch runner neither drains queued work nor rolls back prior completed batches. `batch.completed` counts only load calls that returned successfully, so an observer can distinguish confirmed completed batches from the current uncertain batch. Recovery is a new run plus business idempotency or ready-skip logic.

## Defer concurrent and durable orchestration

The current package deliberately excludes cross-batch pipelining, multiple simultaneously active stages, automatic checkpoint/resume, retry policy, heartbeat, ETA, retention, scheduling, queues, database ledgers, dashboards, arbitrary logging, artifact management, DAGs, streaming records, and exactly-once claims.

Cross-batch pipelining may improve throughput, but it would make a single `stage` field false because transform and load could be active simultaneously. It requires a separate observation design rather than a small optimization to `BatchETLPipeline`. A future ETL DAG remains a parallel class inside `quantmind.etl`, sharing only proven package-private recording mechanics and never inheriting either current pipeline class.
