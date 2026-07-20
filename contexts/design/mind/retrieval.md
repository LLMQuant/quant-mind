# Build and retrieve from a self-contained structure tree

## Quick Summary

- **Purpose**: Define how a paper becomes a self-contained page-preserving
  structure tree, how that tree is persisted and reopened, and how a reasoning
  agent retrieves evidence from it without embeddings replacing reasoning.
- **Read when**: Designing or changing PageIndex-style tree construction, the
  `PaperFlow` pipeline collection, the `mind` retrieval operation, structure-tree
  persistence, or future hybrid semantic-plus-agentic retrieval.
- **Status**: Redesigned (2026-07). Supersedes the earlier "vectorless empty
  shell + library page refill" design. The structure tree now carries its own
  node content; retrieval is a pure value operation over that tree; the library
  dumps and loads the tree as an independent artifact. Semantic node projections
  and hybrid seeding remain deferred.
- **Core rule**: **Pipelines produce complete, self-contained artifacts;
  persistence and retrieval are separate downstream concerns.** A structure
  tree's nodes carry their own text (and optionally an embedding), so the tree
  is usable the moment a pipeline returns it — in memory, with no library. The
  library only dumps and loads that value; single-tree retrieval reasons over
  the tree value and returns **evidence values**, never a bare reference that
  forces a library round-trip. Retrieval reasons over titles and summaries;
  embeddings are a coarse pre-filter added later, never a replacement.
- **Canonical models**: [Paper source and artifact design](../knowledge/paper.md).
- **Related work**: issues #122 (context), #95 (feature request), #71 (`mind`
  scaffold), #120 (source-first Paper Flow V1); prior art `VectifyAI/PageIndex`
  (MIT).

## Contents

- [Motivation](#motivation)
- [Pipelines vs Components (Altitude)](#pipelines-vs-components-altitude)
- [Design at a Glance](#design-at-a-glance)
- [Ownership](#ownership)
- [Self-Contained Structure Tree](#self-contained-structure-tree)
- [The PaperFlow Pipeline Collection](#the-paperflow-pipeline-collection)
- [Persistence: Dump / Load Symmetry](#persistence-dump-load-symmetry)
- [Retrieval Returns Values, Not References](#retrieval-returns-values-not-references)
- [Callable Shape](#callable-shape)
- [Future Multi-Document Composition](#future-multi-document-composition)
- [Multi-Model Compatibility](#multi-model-compatibility)
- [Hybrid Search Compatibility](#hybrid-search-compatibility)
- [Boundaries and Import Contracts](#boundaries-and-import-contracts)
- [Verification Slice](#verification-slice)
- [Out of Scope](#out-of-scope)

## Motivation

Vector retrieval assumes the passage most similar to a query in embedding space
is the most relevant one. For long, structured financial documents that
assumption breaks: near-identical passages differ on a threshold or exception;
fixed-size chunking fragments a table; a cross-reference such as "see Item 7A"
shares no similarity with its target; and a stateless retriever cannot use prior
reasoning to decide where to look.

Reasoning-based retrieval reframes the problem as relevance classification over a
document's real structure: an agent reads a tree of section titles and
summaries, picks a branch, drills down, and reads leaf text with exact page
provenance. `quantmind.knowledge` already records this as the purpose of
`TreeKnowledge`, and embeddings there "act as a coarse pre-filter, never as a
replacement for that reasoning."

The earlier build of this feature made the tree an *empty shell*: nodes stored
only titles, summaries, and page citations, and leaf text was refilled at query
time from the library via `resolve(locator)`. That coupled two things that
should be independent — the tree artifact and the store — and made single-tree
retrieval impossible without a library. This design removes that coupling.

## Pipelines vs Components (Altitude)

This feature is the worked example for the repository's altitude rule (see
[operations/orchestration.md](../operations/orchestration.md)). Two layers, two
jobs:

- **Pipelines** (`quantmind.flows`) are finished, batteries-included workflows.
  A caller states an intent — "turn this paper into a structure tree" — and gets
  back a **complete, self-contained artifact**. A pipeline is pure processing:
  `input → artifact`. It does **not** bind a library, persist anything, or
  retrieve. Producing the artifact fully — including any embeddings it carries —
  is the pipeline's job.
- **Components** (`knowledge`, `preprocess`, `rag`, `library`, `mind`) are the
  building blocks. A caller who wants only an intermediate (just the parsed
  source, just chunks) uses a component directly and wires it themselves. A
  half-finished intermediate is **not** promoted to a public pipeline.

Persistence (`library`) and retrieval (`mind`) are **downstream** of a pipeline,
not part of it. A structure tree returned by `PaperFlow(cfg).build(input)` is
immediately usable; putting it in a library and retrieving from it are separate,
optional steps a caller chooses.

## Design at a Glance

The build spine is solid; the dotted branch is the later hybrid path that adds
embeddings. Processing (including any embeddings) belongs to the pipeline;
`library` only stores and loads; `mind` only retrieves.

```mermaid
flowchart TD
    IN["PaperInput (arxiv / url / local pdf)"]
    subgraph FLW["flows — PaperFlow(cfg).build(input), pure processing"]
        OPEN["fetch + parse the input"]
        OUT["outline signals (deterministic)"]
        DRAFT["draft structuring agent (model, private draft)"]
        BUILD["mint ids/links, read node text from cited pages, validate; self-contained PaperStructureTree (+ as_of / provenance)"]
    end
    subgraph LIB["library — dump / load only"]
        PUT["put(tree): serialize self-contained tree"]
        OPENT["open_structure(id): deserialize same value"]
    end
    subgraph MND["mind — retrieve (pure value op, no library)"]
        RET["AgenticRetriever(cfg).retrieve(tree, question): agentic, evidence values"]
    end
    IN --> OPEN --> OUT --> DRAFT --> BUILD
    BUILD -->|in-memory, use immediately| RET
    BUILD -.->|optional persist| PUT
    OPENT -.->|reopened, identical use| RET
    BUILD -.->|P2 node projections embeddings| PUT
```

## Ownership

| Owner | Responsibility |
|---|---|
| `quantmind.preprocess` | Emit deterministic outline signals (heading candidates, table-of-contents pages, printed-to-physical page offset) from a parsed document. No LLM calls. |
| `quantmind.flows` (`PaperFlow`) | A **config-bound** flow: `PaperFlow(cfg)` binds the settings; `build(input)` fetches, parses, runs one draft-structuring agent, then calls the knowledge constructor and returns a **self-contained** `PaperStructureTree`. The cfg *type* selects the knowledge shape (`PaperStructureCfg` → tree today). No persistence, no retrieval, no library. |
| `quantmind.knowledge` | Own the `StructureTree` structural base and the source-bound `PaperStructureTree` artifact. `from_draft` mints identity, resolves page citations, **and populates each node's `content` from the exact source pages**, then runs the integrity gate. The artifact is a complete value. |
| `quantmind.library` | Dump a self-contained tree and load it back unchanged (`put` / `open_structure`). A tree is an **independent** artifact: its library need not contain a chunk set, and loading it never depends on refilling text from another artifact. |
| `quantmind.mind` | `AgenticRetriever(cfg)` binds the strategy config; `retrieve(tree, question)` reasons over one explicit tree value and returns evidence values with content already in them. It does **not** take or bind a library. |

`quantmind.rag` is unchanged: deterministic chunking and BM25, no LLM, no
PageIndex producer.

## Self-Contained Structure Tree

`StructureTree` stays the shared structural base (a plain `BaseModel`:
`root_node_id`, `nodes`, the `root()` / `children_of()` / `walk_dfs()` /
`find_path()` traversal surface, and the `validate()` integrity gate). Identity
is added by subclasses.

The change is in what a node holds. `TreeNode` already has a
`content: str | None` field; the previous design **forbade** populating it
(`PaperStructureTree` raised "nodes must not copy content"). That rule is
**inverted**:

- A `PaperStructureTree` leaf node **carries its own `content`** — the text of
  the physical source pages it cites, read from the exact source revision at
  build time. The tree is self-contained: it no longer depends on the source
  revision or a chunk set to yield node text.
- The tree also carries the **minimal provenance metadata** it needs to be
  stored and time-queried standalone: `as_of` (financial time, copied from the
  source revision) plus a light source ref (uri / title) and the source content
  hash. This metadata is **not** part of `id` / `content_hash` — a rebuild at a
  different wall-clock time yields the same identity. It is shared via a light
  provenance base (an `ArtifactMeta`-style mixin), **not** full `BaseKnowledge`:
  a structure tree is a derived artifact, not canonical knowledge. With it,
  `library.put(tree)` stores the tree with no other object required.
- A node **may** additionally carry an optional `embedding` (reserved for the
  later hybrid path). Pure agentic retrieval does not need it; when present it is
  produced by the pipeline and persisted with the tree, never computed at
  `library.put` time.
- `content_hash` covers node content (it hashes the full node dump), so a
  self-contained tree versions by its content as expected; the provenance
  metadata stays outside it.
- The redundancy is deliberate and accepted: a structure tree is a derived,
  rebuildable artifact; carrying its own text and light provenance in a local
  knowledge base is worth far more (self-contained, independently retrievable,
  dump/load symmetric) than the storage saved by an empty shell.

Page ranges still reuse `Citation`: a node spanning pages 5-8 carries four
`Citation(page=5..8)` entries. The integrity gate still requires single-rooted,
acyclic, fully reachable topology; bidirectional parent/child links; unique
sibling positions; every cited page inside the source; and every child's pages
contained in its parent's. The only relaxed rule is the content prohibition.

A future document type adds its own `StructureTree` subclass with its own source
binding and its own content-population step; nothing paper-specific leaks into
the base.

## The PaperFlow Pipeline Collection

`PaperFlow` is a **config-bound** flow: you bind the settings once at
construction, and `build(input)` applies them to each input. The cfg **type**
selects which knowledge shape it produces — a paper projects into several
knowledge shapes (tree / flatten / graph), and none is privileged over the
others.

```python
from quantmind.flows import PaperFlow, batch_run
from quantmind.configs import PaperStructureCfg

tree_flow = PaperFlow(PaperStructureCfg(model="gpt-4o-mini"))   # bind cfg once
tree = await tree_flow.build(input)                             # -> PaperStructureTree

# a batch runs every input under one fixed, reproducible setting:
trees = await batch_run(tree_flow.build, inputs)
```

Rules:

- **Bind the config, not the input.** `PaperFlow(cfg)` stores the immutable cfg;
  `build(input)` takes only the operand. A batch therefore runs every input under
  one unified setting, and a config can never drift mid-run — a reproducibility
  requirement, not ergonomics.
- **The cfg type picks the shape.** `PaperStructureCfg` → `PaperStructureTree`
  (implemented now). A later `PaperCardCfg` → the chunk/summary shape reached
  through the same `build` seam; the existing `paper_flow(input, *, cfg)` function
  stays as a thin compatibility wrapper for that semantic shape until it lands.
- **Pure processing.** `build` fetches, parses, and structures, producing the
  complete self-contained artifact. It binds **no** library, persists nothing,
  and does not retrieve.
- A caller who wants only the parsed source uses `quantmind.preprocess` directly;
  "parse only" is a component seam, not a public pipeline.

## Persistence: Dump / Load Symmetry

Because the tree is a self-contained value with its own provenance metadata, the
library reduces to dump/load and needs no second object:

```python
tree_flow = PaperFlow(PaperStructureCfg())
tree = await tree_flow.build(input)              # in-memory, immediately usable

await library.put(tree)                          # dump: nodes, text, embeddings, meta
tree2 = await library.open_structure(tree.id)    # load: identical value object
```

- `library.put(tree)` serializes the whole self-contained tree (structure, node
  text, any node embeddings, and the `as_of` / source-ref provenance). It
  requires **no** source revision or chunk set. Storing the raw source is a
  separate, independent `library.put(source)` when a caller wants the PDF kept.
- `library.open_structure(tree_id)` returns the same `PaperStructureTree` value,
  embeddings and metadata included.
- The library is one store holding **independent** artifacts side by side —
  `PaperSourceRevision`, `PaperChunkSet`, `PaperStructureTree`. A tree does not
  imply a chunk set or a stored source, and vice versa.
- The old `resolve(locator)` **text-refill** path for structure-tree nodes is
  gone; a resolved node returns its own stored content. `resolve` remains for
  cross-document reference resolution (see below), but single-tree retrieval
  never uses it.

## Retrieval Returns Values, Not References

Single-tree retrieval is a **pure value operation** over a self-contained tree.
`AgenticRetriever(cfg)` binds the retrieval config; `retrieve(retrievable, question)`
takes only the operand — a `Retrievable` (a `StructureTree` today; it widens
within `knowledge` to other reasoning-able shapes such as a graph, never to a
vector store):

```python
from quantmind.mind import AgenticRetriever
from quantmind.configs import RetrievalCfg

retriever = AgenticRetriever(RetrievalCfg(model="gpt-4o-mini"))
evidence = await retriever.retrieve(tree, "What are the method and limitations?")
for item in evidence:
    print(item.title, item.content)     # content is already here; no library
```

`RetrievalEvidence` carries the **value**, with the reference as an optional
provenance field, not the path to the content:

```python
class RetrievalEvidence(BaseModel):
    title: str
    content: str                            # value: self-contained, ready to use
    citations: tuple[Citation, ...]         # value: provenance to source pages
    locator: ArtifactLocator | None = None  # optional: for cross-artifact fusion
```

Why this shape resolves the reference dilemma: if retrieval returned a bare node
**reference**, the consumer would have to resolve it against a library, making
every single-tree retrieval library-dependent. Because the tree is
self-contained, retrieval reads `tree.nodes[id].content` directly and returns
it. The `locator` is there for when a caller *wants* to fuse or re-open across
artifacts — an optional capability, never the only way to see content.

There is one strategy: **agentic traversal**. The retriever exposes two SDK
`@function_tool` functions — `get_document_structure()` (tree without leaf text)
and `get_node_content(node_ids)` (leaf text read from `tree.nodes`, **not** a
library) — and lets an Agent decide, turn by turn, which node to open. Content
for a selected non-leaf node is assembled from its descendant leaves. Mechanical
retrieval (semantic vector search / BM25) is a different layer entirely
(`library.search` / `rag`); `mind` never re-implements it, and a future hybrid
path composes the two rather than adding a "semantic strategy" here.

`retrieve` calls `agents.Runner.run(...)` with its own `RunConfig`; it does not
import `flows._runner`. Serialization is bounded by a structure token budget.
Seeds (when a hybrid shortlist eventually supplies them) are **in-tree node
ids**, validated against `tree.nodes`; they carry no library dependency.

## Callable Shape

Both public callables **bind their cfg** and take only the operand per call,
following the repository's altitude rule
([operations/naming.md](../operations/naming.md)):

- `AgenticRetriever(cfg)` is a **class**, not a function. It binds the retrieval
  config so a batch of queries runs under one fixed setting — reproducibility, not
  ceremony. (The earlier `StructureRetriever` bound a *library*; that binding is
  gone now that the tree is self-contained, but the *config* binding remains, so
  the class stays — an absent external dependency is not a reason to demote it.)
  It has a **single behavior** — an LLM agent reasons over the structure — so it
  does no cfg-type dispatch. That is deliberate: `mind` is the reasoning layer,
  and mechanical retrieval (semantic / BM25) stays in `library` / `rag`, never a
  second strategy bolted onto this class.
- `PaperFlow(cfg)` is a **class** for the same reason: it binds the build config
  once so a batch builds every input identically, and its cfg type selects the
  knowledge shape. It binds no source, no store, and no per-call state.

## Future Multi-Document Composition

References and the library earn their keep at **corpus scale**, which is exactly
where single-tree value retrieval stops fitting in memory:

1. select candidate documents using metadata, descriptions, or global-summary
   projections;
2. `library.open_structure(...)` each candidate tree by identity;
3. reuse one `AgenticRetriever(cfg).retrieve(tree, question)` per explicit tree;
4. fuse evidence using the full `(source_revision_id, artifact_id, node_id)`
   locator carried on each `RetrievalEvidence`.

Here the `locator` field and library-backed `resolve` are the right tools:
you cannot hold thousands of trees in memory, so you shortlist by reference and
open on demand. The current release does not implement the collection router or
evidence fusion; the seam is the optional `locator` on evidence plus
independent per-artifact persistence.

## Multi-Model Compatibility

- **Rely on the SDK.** `cfg.model` (a plain string, including a
  `litellm/<provider>/<model>` value) flows unchanged into the SDK `Runner`.
- **Capability requirement.** Draft structuring needs reliable structured output;
  agentic traversal needs tool-calling. A provider lacking a stage's capability
  is unsupported for that stage. Tests cover at least one non-OpenAI model.
- **Embeddings.** The later hybrid step depends only on the library's existing
  embedding seam and on `SemanticQuery` / `SemanticHit`, never on a specific
  vendor.

## Hybrid Search Compatibility

Hybrid retrieval — shortlist nodes by semantic search, then reason over the
shortlist — is a later, explicit step. Node embeddings are produced **by the
pipeline** and persisted **with the tree** (not computed at `put` time), so the
tree stays self-contained end to end. Two independent embedding sets can then
coexist in one library — chunk-set embeddings and structure-node embeddings —
and hybrid queries both. Seeds remain validated in-tree node ids; a hit never
becomes an answer without the reasoning step.

## Boundaries and Import Contracts

- `library` and `rag` must not import `quantmind.mind`.
- `mind` may import `knowledge`, `configs`, and `utils`. It **no longer needs to
  import `library`** for retrieval: single-tree `retrieve` is library-free. The
  existing `mind -> library` allowance may stay for the future collection path,
  but the retrieval operation must not depend on it.
- `flows` is the apex and may import `knowledge`, `preprocess`, `rag`, `configs`,
  `mind`, and `utils`. `PaperFlow` imports no library.
- Update the `import-linter` contracts if the `mind -> library` edge is dropped
  from the retrieval path; keep every other edge intact.

## Verification Slice

Offline tests use fixed PDFs and fake model outputs. They cover: a table of
contents, a missing table of contents, a printed page-number reset, and an
in-body cross-reference; every tree-integrity rejection; **node content
populated from cited pages and preserved through dump/load**; stable IDs and
idempotent re-runs; a tree persisted and reopened as an identical self-contained
value **without any chunk set present**, with `as_of` / provenance preserved;
agentic retrieval returning evidence **whose content comes from the tree, with
no library involved**; a bound `PaperFlow(cfg)` producing a self-contained tree
with the cfg *type* selecting the shape; `AgenticRetriever(cfg)` binding its
config; multi-model identity forwarding; and in-tree seed validation. P2 adds
seeded semantic-shortlist tests once node projections exist.

## Out of Scope

- an empty-shell tree or a query-time text-refill path;
- a nested `TreeKnowledge` inside an artifact, or a second parallel tree
  implementation;
- a `Citation.end_page` field or a separate page-resolver concept;
- a shared runtime module or moving `flows._runner`;
- a generic retriever hierarchy, vector-store abstraction, provider registry, or
  query-engine hierarchy (a single `retrieve` that branches on cfg/kind is not
  this);
- answer synthesis or agent memory inside the retrieval primitive;
- collection routing or evidence fusion in this release;
- knowledge-graph construction.
