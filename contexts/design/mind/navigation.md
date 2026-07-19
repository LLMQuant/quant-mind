# Build and navigate page-preserving knowledge trees

## Quick Summary

- **Purpose**: Define how QuantMind builds a page-preserving, source-linked
  navigation tree from a source-first paper and how a reasoning agent navigates
  it, without embeddings replacing that reasoning.
- **Read when**: Designing or changing PageIndex-style tree construction, the
  `mind` navigation package, the shared agent runtime seam, hybrid
  semantic-plus-agentic retrieval, or the multi-provider model matrix.
- **Status**: Planned. No implementation exists on `master`. This page records
  the accepted design and refines issue #95 for the source-first Paper Flow V1
  model shipped in #120; the `quantmind.mind` package does not exist yet.
- **Core rule**: A navigation tree is a source-linked, independently versioned
  artifact with code-owned identity, links, and citations. The model proposes a
  draft; code validates every node. Navigation reasons over titles and
  summaries; embeddings are a coarse pre-filter, never a replacement.
- **Related work**: issues #122 (context), #95 (feature request), #71 (`mind`
  scaffold), #111 (`LocalKnowledgeLibrary`, closed), #120 (source-first Paper
  Flow V1); prior art `VectifyAI/PageIndex` (MIT).

## Contents

- [Motivation](#motivation)
- [Non-Goals](#non-goals)
- [Ownership](#ownership)
- [Artifact Identity](#artifact-identity)
- [Tree Model](#tree-model)
- [Build Pipeline](#build-pipeline)
- [Navigation Retrieval](#navigation-retrieval)
- [Shared Agent Runtime](#shared-agent-runtime)
- [Multi-Model Compatibility](#multi-model-compatibility)
- [Hybrid Search Compatibility](#hybrid-search-compatibility)
- [Boundaries and Import Contracts](#boundaries-and-import-contracts)
- [Phasing](#phasing)
- [Acceptance and Evaluation](#acceptance-and-evaluation)
- [Relationship to Prior Art](#relationship-to-prior-art)
- [Out of Scope](#out-of-scope)

## Motivation

Vector retrieval assumes the passage most similar to a query in embedding space
is the most relevant one. For long, structured financial documents (10-K/10-Q
filings, prospectuses, research reports) that assumption breaks in ways that
matter: near-identical passages differ on a threshold, condition, or exception;
fixed-size chunking fragments a table or a clause; a cross-reference such as
"see Item 7A" shares no semantic similarity with its target; and a stateless
retriever cannot use prior reasoning or conversation to decide where to look.

Reasoning-based navigation reframes retrieval as relevance classification
performed by the model over a document's real structure. The agent reads a tree
of section titles and summaries, picks a branch, drills down, and lazily loads
leaf text with exact page provenance. `quantmind.knowledge` already records this
as the purpose of `TreeKnowledge`: "an agent reads the root summary plus
children summaries, picks the most likely branch, drills down, and lazy-loads
leaf content. Embeddings (when available) act as a coarse pre-filter, never as a
replacement for that reasoning."

This page defines how that capability is built and owned across packages so the
pieces compose without a new framework.

## Non-Goals

Carried from issue #95 and the existing library and RAG boundaries:

- A general-purpose RAG or PageIndex framework, provider registry, or retriever
  hierarchy.
- A second tree schema (`PageIndexNode`) or a generic `Document` model. The tree
  payload reuses `TreeKnowledge`, `TreeNode`, and `Citation`.
- A second persistence layer or semantic index. Persistence and semantic
  candidates stay in `LocalKnowledgeLibrary`.
- Answer synthesis inside the navigation primitive. Navigation returns node
  evidence; a caller composes an answer.
- Making `Paper` a `TreeKnowledge` again. Paper Flow V1 deliberately has no
  paper tree; a navigation tree is a separate, source-linked artifact.

## Ownership

The feature is cross-cutting. Each existing package keeps its current
responsibility; the agentic traversal introduces a new owner, `mind`, and the
shared agent runtime moves below `flows`.

| Owner | Responsibility |
|---|---|
| `quantmind.preprocess` | Preserve page boundaries and emit deterministic outline signals (heading candidates, table-of-contents pages, printed-to-physical page offset). No LLM calls. |
| `quantmind.rag` | Optional stateless helper: propose a draft outline from one `ParsedDocument`. Returns a bounded draft, never canonical identity. Imports only `preprocess`. |
| `quantmind.knowledge` | Provide the `TreeKnowledge` / `TreeNode` / `Citation` payload models and the `PaperNavigationArtifact` envelope with its validation rules. No I/O, no retrieval-text choice. |
| `quantmind.flows` | Turn a model draft into a validated canonical artifact linked to an exact `PaperSourceRevision` and chunk set. Persistence stays explicit. |
| `quantmind.library` | Persist the canonical artifact through the paper artifact tables and, as an explicit later step, build per-node projections. Adds no second store or index. |
| `quantmind.mind` | Own the shared agent runtime seam and perform agentic traversal, returning node evidence. May request a semantic shortlist from `library`. |

This resolves a prior tension. `contexts/design/library/local.md` and
`contexts/design/rag/document.md` say a future PageIndex operation "may live
under `quantmind.rag`". That holds only for the stateless, document-local draft
producer, because `rag` may import only `preprocess`. Building a persisted
canonical artifact and running hybrid, library-backed navigation sit above
`library` and belong to `flows` (build) and `mind` (navigate).

## Artifact Identity

A navigation tree is a source-first artifact, not a bare `TreeKnowledge`. A
plain `TreeKnowledge` cannot carry the identity the paper design requires: its
`id` defaults to a random UUID, `ExtractionRef` records no producer
configuration or input artifact, and `knowledge_items` has no link to
`paper_sources` or `paper_artifact_lineage`. Persisting it with the conventional
`put()` would make re-runs non-idempotent, allow a deleted source to orphan a
tree, and hide the tree from `get_paper()`.

The design therefore adds a `PaperNavigationArtifact` envelope, mirroring
`PaperChunkSet` and `PaperGlobalSummary`:

- a new `PaperArtifactKind` value (for example `paper_navigation_tree`);
- `source_revision_id` binding it to an exact `PaperSourceRevision`;
- a `producer` config (model, prompt version, input chunk-set id, instructions
  hash, structuring bounds) and a `producer_config_hash`;
- a `content_hash` over the canonical tree;
- lineage to the input `PaperChunkSet` via `paper_artifact_lineage`;
- a stable, content-derived artifact id, so an unchanged re-run is idempotent and
  a changed producer configuration versions rather than overwrites.

The envelope reuses `TreeKnowledge` / `TreeNode` as its payload; it does not
introduce a second tree schema. It is persisted through the existing
`paper_artifacts` / `paper_artifact_members` / `paper_artifact_lineage` tables
(nodes as members), never `knowledge_items`, so identity, lineage, and
fail-closed rehydration match every other paper artifact.

## Tree Model

- `TreeNode` already carries `node_id`, `parent_id`, `position`, `title`,
  `summary`, optional `content`, `citations`, and `children_ids`. A section maps
  to a node; the ordered PDF pages it spans map to node `citations`.
- Page provenance uses `Citation`. Add an optional `Citation.end_page`:
  `Citation.page` is the 1-based inclusive start (constrained `>= 1`), an omitted
  end means one page, a present end must satisfy `end_page >= page`, an end with
  no start is rejected, and both must fall within the source page count.
- A node's `content` and page citations reuse already-validated `PaperChunk`
  text and `PaperSourceSpan` page numbers rather than re-parsing the PDF, so a
  leaf never invents text or a page.

## Build Pipeline

Construction separates deterministic work from model work, and keeps canonical
identity, links, and content in code rather than in model output.

1. **Outline signals (`preprocess`, deterministic).** From the page-aware
   `ParsedDocument`, detect table-of-contents pages, heading candidates, and the
   printed-to-physical page offset. Emit ordered, page-anchored signals; make no
   LLM call and no ranking decision.
2. **Draft structuring (`flows`, Agents SDK).** An agent proposes a *private
   draft* hierarchy from outline signals and chunk text: titles, nesting, and
   per-node draft summaries with candidate page ranges. The model output is never
   canonical; it is the equivalent of `PaperCitationDraft`.
3. **Canonicalization (`flows`, code-owned).** Code assigns node ids, builds
   parent/child links, resolves each leaf's `content` and `citations` from the
   `PaperChunkSet`, and assembles the `PaperNavigationArtifact` with its producer
   identity and lineage.
4. **Full integrity validation (`knowledge` + `flows`, deterministic).** Reject
   any artifact that is not a single-rooted, acyclic tree with: every node
   reachable from the root; bidirectional parent/child consistency; unique
   sibling positions; no orphan or unreachable node; every citation range inside
   the source page count; and every child page span contained in its parent's
   span. This is the integrity gate. Title-appearance sampling is retained only
   as a quality signal, not as the gate, and a low quality score falls back to a
   flat single-level tree rather than emitting an unverified hierarchy.
5. **Persistence (`library`).** Store the canonical `PaperNavigationArtifact`
   through the paper artifact tables. Persistence does not require embeddings;
   projections are a separate, explicit step (see
   [Phasing](#phasing)).

Steps 1, 3, and 4 hold no model dependency, so page, link, and content
correctness are testable without a network. Step 2 is the only stage that calls
a model.

## Navigation Retrieval

Navigation lives in `quantmind.mind.navigation` and returns node evidence
(locators, titles, page-cited content), never a synthesized answer. Its inputs
are explicit, so every tool it exposes is implementable:

```text
navigate(artifact, question, *, page_resolver, seed_locators=None)
  -> list[NavigationEvidence]
```

- `artifact` is the `PaperNavigationArtifact` (tree payload plus identity, so
  seed locators can be validated against it).
- `page_resolver` resolves node or page content from the exact
  `PaperSourceRevision` / `PaperChunkSet`. The tree is not page storage; leaf
  text comes from the resolver, matching how the reference implementation reads
  cached pages rather than reconstructing them from the tree.

Two grains are supported:

- **Single-pass selection.** Serialize the tree with node text stripped (ids,
  titles, summaries, hierarchy only), make **one** model call for the relevant
  node ids, then load exact page-cited content for those nodes through the
  resolver in code. Cheap and predictable.
- **Agentic traversal.** Expose SDK `@function_tool` functions —
  `get_document_structure()` (tree without leaf text), `get_children(node_id)`
  (progressive expansion for large trees), and `get_node_content(node_ids)`
  (exact page-cited leaf text via the resolver) — and let an Agent decide, turn
  by turn, which branch to open and when it has enough evidence. This path can
  follow cross-references and use conversation state, at variable model cost.

Serializing the whole structure is bounded by node count, not depth, so large
trees use a structure token budget plus `get_children()` progressive expansion
rather than one giant blob. Both grains run through the shared runtime seam (see
below) for tracing and turn limits.

## Shared Agent Runtime

Navigation must reuse the observability run wrapper and the model seam that
`flows` uses, but `mind` may not import `flows`. The shared runtime therefore
does not live in `flows`.

Decision: place the run wrapper and model resolution in a shared module that both
layers import downward — `quantmind.mind.runtime` (with `flows` depending on
`mind`, which the contracts already permit) or a neutral `quantmind.runtime`
module. The existing `flows._runner` moves there. `flows` and `mind.navigation`
both call the shared runner; neither owns a private copy. This choice couples
with the #71 `mind` scaffold and is a prerequisite for P1.

## Multi-Model Compatibility

The design must not hard-lock to one provider for generation or embeddings. On
`master` today `cfg.model` is a plain string passed straight to
`agents.Agent(model=...)`, and the only embedding provider is OpenAI.

- **Rely on the SDK, do not wrap it.** The Agents SDK already routes a
  `litellm/<provider>/<model>` model string through its multi-provider support,
  so the design adds no bespoke provider-resolution wrapper. `cfg.model` flows
  unchanged into the shared runtime seam.
- **Define a capability contract instead.** Each stage states what it needs from
  a model: draft structuring needs reliable structured output; agentic traversal
  needs tool-calling; all stages need usage reporting for the cost caps in
  `BaseFlowCfg`. A provider that cannot meet a stage's contract is unsupported
  for that stage, and this is covered by a provider test matrix (at least one
  OpenAI and one non-OpenAI model across structuring and traversal), not by a
  runtime abstraction.
- **Provider-agnostic embeddings.** The library isolates embeddings behind the
  private `_EmbeddingProvider` Protocol. Hybrid navigation depends only on that
  seam and on `SemanticQuery` / `SemanticHit`, never on a specific vendor.

## Hybrid Search Compatibility

Hybrid retrieval — use semantic search to shortlist nodes, then let the agent
reason over that shortlist — is a first-class later mode. It reuses locator
identity end to end.

- After a navigation artifact's nodes are projected (the explicit P2 step),
  `search(SemanticQuery(...))` over those projections returns `SemanticHit`
  values. Each hit already carries a full `ArtifactLocator` (source revision,
  artifact id, artifact kind, member id); the design keeps the locator and does
  not collapse a hit to a bare `node_id`.
- Seeds are locators, not node ids: `navigate(..., seed_locators=hits.locator)`.
  Single-tree mode constrains the query with
  `SemanticQuery(tree_id=artifact.id, artifact_kinds=[paper_navigation_tree])`
  and rejects any seed whose artifact id does not match the artifact under
  navigation, so a hit from another tree, a flat item, or a chunk cannot leak in.
- Corpus mode (P3) navigates across artifacts and therefore keys seeds by the
  full `(artifact_id, node_id)` locator, not node id alone.
- Embeddings stay a coarse pre-filter. The agent may leave the seeded subtree,
  and a hit never becomes an answer without the reasoning step.

Pure-agentic navigation passes no seeds; hybrid passes validated locator seeds.
Both share one primitive; hybrid adds a shortlist step in front of it.

## Boundaries and Import Contracts

Placement must satisfy the `import-linter` contracts. Relevant directions:
`knowledge` is a leaf; `library` imports only `knowledge`; `rag` imports only
`preprocess`; `flows` + `magic` is the apex, and may import `mind`.

- When `mind` is implemented, add a contract pinning `mind -> library ->
  knowledge`: `mind` may import `knowledge`, `library`, `configs`, and `utils`,
  but not `flows`, `magic`, `rag`, or `preprocess`. `library` and `rag` remain
  barred from importing `mind`.
- The shared runtime that both `flows` and `mind` use lives at or below `mind`
  (see [Shared Agent Runtime](#shared-agent-runtime)); the contract must permit
  `flows -> mind` and forbid `mind -> flows`.
- The stateless draft helper stays in `rag` and keeps `rag`'s
  imports-only-`preprocess` rule intact.
- Canonical artifact construction stays in `flows`.

The existing `## PageIndex Boundary` (library) and `## Collection Search and
PageIndex` (rag) sections cross-link this page; agentic, library-backed
navigation is owned by `mind`, while `rag` remains draft-only.

## Phasing

Each phase is independently shippable and testable, and only P2 introduces
embeddings.

- **P0 — Build and persist, vectorless.** Deterministic outline signals, the
  `Citation.end_page` addition, the `PaperNavigationArtifact` with producer
  identity and lineage, code-owned canonicalization and full integrity
  validation, and library persistence **without** projections. No embedding
  provider is required.
- **P1 — Agentic navigation, vectorless.** The shared runtime relocation and the
  `mind.navigation` primitive with single-pass and agentic traversal over the
  persisted artifact, returning node evidence. Still no embeddings.
- **P2 — Semantic projections and hybrid.** Build per-node projections as an
  explicit step, then seed navigation from validated `search()` locators. This is
  the first phase that needs an embedding provider.
- **P3 — Corpus navigation.** Extend seeds and traversal across artifacts using
  full locators. Design-only here; it must not add a provider registry or a
  second store.

## Acceptance and Evaluation

Implementation is accepted against measurable criteria, not just green unit
tests:

- **Structure fixtures**: documents with a clean table of contents, with no
  table of contents, with a printed page-number reset (roman front matter), and
  with in-body cross-references. Each must build a valid, fully validated tree or
  a declared flat fallback.
- **Citation accuracy**: sampled node citations resolve to the correct source
  pages above a stated accuracy floor.
- **Evidence recall**: for a labelled question set, the fraction of gold evidence
  nodes returned by navigation.
- **Cost and latency**: model calls, tokens, and wall-clock for build and per
  query, reported rather than assumed.
- **Comparison**: pure-agentic vs hybrid vs the existing vector-search baseline
  on the same question set, so the mode choice is evidence-backed.

## Relationship to Prior Art

`VectifyAI/PageIndex` (MIT) validates the approach: table-of-contents detection,
recursive node subdivision, a node schema of title / id / page-range / summary /
children, and two retrieval paths (single-pass node selection and an Agents-SDK
tool loop). QuantMind reuses these mechanics where useful but keeps its own
artifact identity, financial-time, provenance, and citation contracts, and
treats PDF content as untrusted input to structuring calls. Reuse of PageIndex
code remains subject to license, dependency, and contract review; commodity
mechanics are not reimplemented before that review.

A related internal system (FinMind / SmartAsk) validated a two-level
document-then-page navigation with two-step model selection and measured gains
over an embedding baseline on FinanceBench. This design generalizes that to a
verified multi-level tree with exact page-range citations and a hybrid pre-filter
rather than a fixed two levels.

## Out of Scope

- A second tree, node, or document schema; a `PaperTree` on `Paper`.
- A public retriever, vector-store, provider registry, or query-engine hierarchy.
- A second persistence or semantic-index layer.
- Answer synthesis or agent memory inside the navigation primitive.
- Corpus-level virtual nodes or query-time tree reconstruction (design only,
  after single-document navigation ships).
- Knowledge-graph construction.
