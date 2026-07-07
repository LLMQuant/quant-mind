# Design: paper_flow draft-schema + code-assembled UUIDs

Date: 2026-07-07
Status: Approved (pending spec review)
Scope: `quantmind/flows/paper.py`, new `quantmind/flows/_paper_draft.py`, tests

## Problem

`paper_flow` could not produce a `Paper` end-to-end against a live model.
Three failure modes were found by actually running it (the test suite mocks
the SDK, so `scripts/verify.sh` stayed green and never caught them):

1. **Strict JSON schema rejected `Paper`.** `Agent(output_type=Paper)` fails
   at `Runner.run` with `UserError: Strict JSON schema is enabled, but the
   output type is not valid` — key-independent. Root cause: `Paper` →
   `TreeKnowledge.nodes: dict[UUID, TreeNode]` maps to an open-ended
   `additionalProperties` object, which strict mode forbids.

2. **The model cannot reliably emit UUIDs.** After wrapping the output type
   non-strict, the model returned semantic ids (`"paper_001"`,
   `"introduction"`) for the many `UUID`-typed fields (`id`, `root_node_id`,
   `node_id`, the `nodes` dict keys, and `Citation.tree_id` / `node_id`),
   which fail Pydantic validation.

3. **Even when told to emit UUIDs, the model corrupts them.** Prompt-patching
   reduced but never eliminated the errors: the model hallucinated an invalid
   hex character (`...-bdle-...` instead of `...-bd1e-...`) and *reused* one
   UUID for two different nodes (a duplicate JSON key that silently drops a
   node — Pydantic does not even flag it).

Conclusion: asking an LLM to hand-generate dozens of unique, valid,
cross-referenced RFC-4122 UUIDs is structurally unreliable. The fix is to
stop the LLM from ever touching an identifier.

## Approach (chosen: A — nested, id-less draft)

Introduce an LLM-facing **draft schema** whose nodes carry only content and
nest directly (no ids, no adjacency lists). Code walks the draft, mints all
UUIDs, wires the tree, and assembles the real `Paper`. Because the LLM never
writes an identifier, id typos, id reuse, and dangling references become
impossible by construction. The draft also uses a `list` (not a `dict`), so
the LLM-facing schema has no open-ended `additionalProperties`.

Rejected alternative B (flat list with string ids): still asks the LLM to own
id consistency; only downgrades UUID errors to string errors instead of
removing the class entirely.

## Components

### New module `quantmind/flows/_paper_draft.py`

LLM-facing schema (all `model_config = ConfigDict(extra="forbid")`):

```python
class DraftCitation(BaseModel):
    quote: str | None = Field(default=None, max_length=500)
    page: int | None = None
    char_offset: int | None = None

class DraftNode(BaseModel):
    title: str
    summary: str
    content: str | None = None
    citations: list[DraftCitation] = Field(default_factory=list)
    children: list["DraftNode"] = Field(default_factory=list)

class DraftPaper(BaseModel):
    title: str
    summary: str
    published_date: date | None = None   # model reads it from the PDF; feeds as_of
    arxiv_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    asset_classes: list[str] = Field(default_factory=list)
    root: DraftNode
```

Assembly (pure, deterministic, no I/O):

```python
def assemble_paper(
    draft: DraftPaper,
    *,
    source: SourceRef,
    source_id: str,
    as_of: datetime,
    extraction: ExtractionRef,
    out_type: type[Paper],
) -> Paper
```

Behaviour:
- Generate `paper_id = uuid4()` up front (needed for `Citation.tree_id`).
- DFS over `draft.root`; for each `DraftNode` mint `uuid4()` and build a
  `TreeNode(node_id, parent_id, position, title, summary, content,
  children_ids, citations)`.
  - `position` = index among its siblings.
  - `children_ids` = the freshly minted uuids of that node's `children`.
  - each `DraftCitation` → `Citation(source_id=source_id, tree_id=paper_id,
    node_id=<this node's uuid>, quote, page, char_offset)`.
- `root_node_id` = root node's uuid; `nodes` = the flat `dict[UUID, TreeNode]`.
- Construct `out_type(id=paper_id, as_of=as_of, source=source,
  extraction=extraction, root_node_id=..., nodes=..., arxiv_id=draft.arxiv_id,
  authors=draft.authors, asset_classes=draft.asset_classes)`. `item_type` is
  left to the model's class default (`"paper"`), not passed explicitly.
- Top-level `Paper.citations` stays empty; citations live on their nodes.

### `quantmind/flows/paper.py` changes

- Agent output type: `AgentOutputSchema(DraftPaper, strict_json_schema=False)`
  (replaces the transitional `AgentOutputSchema(Paper, ...)` wrap).
- `_DEFAULT_INSTRUCTIONS` rewritten to describe the nested draft (title /
  summary / content / citations / children). Remove all UUID guidance and the
  `as_of` / `source` guidance (code owns provenance now). Keep the three cfg
  flags (`extract_methodology`, `extract_limitations`, `asset_class_hint`).
- After `run_with_observability` returns a `DraftPaper`, paper_flow builds
  provenance and calls `assemble_paper(...)`, returning the final `Paper`.
- Provenance derivation (in paper_flow, from `source_meta`):
  - `source: SourceRef` — `kind` + `uri` from the fetch metadata (`arxiv` /
    `web` / `local` / `inline`; `doi` remains NotImplemented). `fetched_at`
    and `content_hash` are left `None` in this MVP (the current `source_meta`
    does not carry them; a follow-up can populate the sha256 dedup hash).
  - `source_id: str` — a stable string (arxiv id / url / path / `"inline"`).
  - `as_of` — `draft.published_date` (as UTC datetime) if present, else
    `datetime.now(timezone.utc)`.
  - `extraction: ExtractionRef(flow="paper_flow", model=cfg.model,
    run_id=None, extracted_at=now)`.
- `output_type` override keeps meaning "final Paper type": the Agent always
  emits `DraftPaper`; `out_type` is used only in `assemble_paper`.

## Data flow

```
input → _fetch_and_format → (markdown, source_meta)
      → Agent(output_type=DraftPaper) via run_with_observability → DraftPaper
      → assemble_paper(draft, source, source_id, as_of, extraction, out_type)
      → Paper
```

## Error handling

- Id typos / reuse / dangling refs are eliminated by construction (LLM never
  emits ids).
- `assemble_paper` is pure and unit-testable; no network, no clock beyond
  `as_of` fallback.
- `DraftPaper.root` is required, so a malformed draft still triggers the SDK's
  normal validation-retry path.

## Testing (TDD)

New `tests/flows/test_paper_draft.py`:
- single-root draft → `Paper` with one node; `root_node_id == node_id`;
  `source` / `as_of` / `extraction` set from args.
- nested draft (root + 2 children + 1 grandchild) → correct `parent_id`,
  `children_ids`, `position`; uuids unique; `walk_dfs` order preserved.
- citations on a node → `Citation.node_id` is that node's uuid,
  `Citation.tree_id == paper_id`, `quote` / `page` preserved,
  `source_id` set.
- `out_type=MyPaper` (a `Paper` subclass) → returns a `MyPaper` instance.

Update `tests/flows/test_paper.py`:
- stub the runner to return a `DraftPaper` (replace `_stub_paper()` with a
  `_stub_draft()` helper) and assert paper_flow returns the assembled `Paper`.
- rewrite `test_output_type_override_propagated` to assert
  `isinstance(result, MyPaper)` after assembly (the Agent's `output_type` is
  now always `DraftPaper`).

## Architecture / boundaries

- `_paper_draft.py` imports only `quantmind.knowledge` (Paper, TreeNode,
  Citation, SourceRef, ExtractionRef) + stdlib. It sits inside the `flows`
  apex; no `import-linter` contract changes.
- Follows repo conventions: Pydantic at the LLM boundary; frozen value types
  (`TreeNode`, `Citation`) constructed internally; functions over classes.

## Out of scope (YAGNI)

- `news_flow` / `earnings_flow` (same pattern can be applied later).
- `PaperKnowledgeCard` generation.
- DOI → OA PDF (unpaywall) resolver — tracked separately.
- Seeded/deterministic uuids — tests assert structure, not specific ids.
```
