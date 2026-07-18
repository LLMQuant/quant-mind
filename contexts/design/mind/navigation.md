# Build and navigate page-preserving knowledge trees

## Quick Summary

- **Purpose**: Define how QuantMind builds a page-preserving, independently
  versioned knowledge tree from a source-first paper and how a reasoning agent
  navigates it, without embeddings replacing that reasoning.
- **Read when**: Designing or changing PageIndex-style tree construction, the
  `mind` navigation package, hybrid semantic-plus-agentic retrieval, or the
  multi-provider model seam these use.
- **Status**: Planned. No implementation exists on `master`. This page records
  the accepted design and refines issue #95 for the source-first Paper Flow V1
  model shipped in #120; the `quantmind.mind` package does not exist yet.
- **Core rule**: Navigation reasons over titles and summaries; embeddings are a
  coarse pre-filter, never a replacement. The tree is a rebuildable derived
  artifact linked to an exact source, never a second canonical identity.
- **Related work**: issues #95 (feature request), #71 (`mind` scaffold), #111
  (`LocalKnowledgeLibrary`, closed), #120 (source-first Paper Flow V1);
  prior art `VectifyAI/PageIndex` (MIT).

## Contents

- [Motivation](#motivation)
- [Non-Goals](#non-goals)
- [Ownership](#ownership)
- [Tree Model](#tree-model)
- [Build Pipeline](#build-pipeline)
- [Navigation Retrieval](#navigation-retrieval)
- [Multi-Model Compatibility](#multi-model-compatibility)
- [Hybrid Search Compatibility](#hybrid-search-compatibility)
- [Boundaries and Import Contracts](#boundaries-and-import-contracts)
- [Phasing](#phasing)
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
performed by the model over a document's real structure. The agent reads a
tree of section titles and summaries, picks a branch, drills down, and lazily
loads leaf text with exact page provenance. `quantmind.knowledge` already
records this as the purpose of `TreeKnowledge`: "an agent reads the root
summary plus children summaries, picks the most likely branch, drills down, and
lazy-loads leaf content. Embeddings (when available) act as a coarse pre-filter,
never as a replacement for that reasoning."

This page defines how that capability is built and owned across packages so the
pieces compose without a new framework.

## Non-Goals

Carried from issue #95 and the existing library and RAG boundaries:

- A general-purpose RAG or PageIndex framework, provider registry, or retriever
  hierarchy.
- A second tree schema (`PageIndexNode`) or a generic `Document` model. The
  design reuses `TreeKnowledge`, `TreeNode`, and `Citation`.
- A second persistence layer or semantic index. Persistence and semantic
  candidates stay in `LocalKnowledgeLibrary`.
- Answer synthesis inside the navigation primitive. Navigation returns node
  evidence; a caller composes an answer.
- Making `Paper` a `TreeKnowledge` again. Paper Flow V1 deliberately has no
  paper tree; a navigation tree is a separate, independently versioned artifact.

## Ownership

The feature is cross-cutting. Each existing package keeps its current
responsibility; only the agentic traversal introduces a new owner, `mind`.

| Owner | Responsibility |
|---|---|
| `quantmind.preprocess` | Preserve page boundaries and emit deterministic outline signals (heading candidates, table-of-contents pages, printed-to-physical page offset). No LLM calls. |
| `quantmind.rag` | Optional stateless helper: propose a draft outline or navigation evidence from one `ParsedDocument`. Returns a bounded draft, never canonical identity. Imports only `preprocess`. |
| `quantmind.knowledge` | Provide the canonical `TreeKnowledge` / `TreeNode` / `Citation` models and page provenance. No I/O, no retrieval-text choice. |
| `quantmind.flows` | Build and enrich the canonical tree with the Agents SDK, linking it to an exact `PaperSourceRevision`. Persistence stays explicit. |
| `quantmind.library` | Persist the canonical tree and provide semantic candidates through existing per-node projections. Adds no second store or index. |
| `quantmind.mind` | Perform agentic traversal over titles and summaries and return node evidence. May request a semantic shortlist from `library`. |

This resolves a prior tension. `contexts/design/library/local.md` and
`contexts/design/rag/document.md` say a future PageIndex operation "may live
under `quantmind.rag`". That remains true only for the stateless,
document-local draft producer, because `rag` may import only `preprocess`. The
stateful parts — building a persisted canonical tree and running hybrid,
library-backed agentic navigation — sit above `library` and therefore belong to
`flows` (build) and `mind` (navigate).

## Tree Model

The navigation tree is a canonical `TreeKnowledge` value, not a new schema.

- `TreeNode` already carries `node_id`, `parent_id`, `position`, `title`,
  `summary`, optional `content`, `citations`, and `children_ids`. A section maps
  to a node; the ordered PDF pages it spans map to node `citations`.
- Page provenance uses `Citation`. `Citation.page` is the 1-based inclusive
  start. Add an optional `Citation.end_page`: an omitted end means one page, and
  a present end must satisfy `end_page >= page`. This is the only canonical-model
  change and matches issue #95.
- The tree is a derived artifact, not a second identity. It sets `source` to the
  same `SourceRef` as the paper it describes, sets `extraction` to the
  building flow and model, and links every node's leaf text back to exact source
  pages. It is independently versioned by its producer configuration, exactly as
  `PaperChunkSet` and `PaperGlobalSummary` are.
- `TreeKnowledge` navigation helpers (`root()`, `children_of()`, `walk_dfs()`,
  `find_path()`) are the traversal surface; the navigation package adds no
  parallel tree structure.

A navigation tree is built from a `PaperSourceRevision` and its `PaperChunkSet`
so that a node's `content` and page citations reuse already-validated chunk text
and spans rather than re-parsing the PDF.

## Build Pipeline

Construction separates deterministic work from model work so page and outline
facts stay reproducible and free of LLM cost.

1. **Outline signals (`preprocess`, deterministic).** From the page-aware
   `ParsedDocument`, detect table-of-contents pages, heading candidates, and the
   printed-to-physical page offset. Emit ordered, page-anchored signals; make no
   LLM call and no ranking decision.
2. **Structuring (`flows`, Agents SDK).** An agent turns outline signals and
   chunk text into a `TreeKnowledge`: section titles, hierarchy, per-node
   summaries, and page-range citations. When no usable table of contents exists,
   the agent proposes a hierarchy directly from content. Oversized sections are
   split by bounded page and token budgets, mirroring PageIndex's recursive node
   subdivision.
3. **Verification (`flows`, deterministic guardrail).** Sample node titles and
   confirm each claimed section actually begins on its cited page against
   `ParsedDocument` evidence. Below an accuracy floor, fall back to a weaker
   structuring mode or a flat single-level tree rather than emit an unverified
   hierarchy.
4. **Persistence (`library`).** `put()` stores the canonical `TreeKnowledge`
   using existing `knowledge_items` / `knowledge_nodes` rows and produces one
   aggregate projection plus one projection for every non-root node. No new
   table is required for the MVP.

Steps 1 and 3 hold no model dependency, so page and outline correctness are
testable without a network. Step 2 is the only provider-dependent stage.

## Navigation Retrieval

Navigation lives in `quantmind.mind.navigation` and returns node evidence
(locators, titles, page-cited content), never a synthesized answer. Two grains
are supported, matching the two paths in the reference implementation.

- **Single-pass classification.** Serialize the tree with node text stripped
  (ids, titles, summaries, hierarchy only), ask the model once for the relevant
  node ids, then load exact page-cited content for those nodes. Cheap and
  bounded: two model calls independent of tree depth.
- **Agentic traversal.** Expose SDK `@function_tool` functions over the tree —
  `get_document_structure()` (tree without leaf text) and
  `get_page_content(pages)` (exact page-cited leaf text) — and let an Agent
  decide, turn by turn, which branch to open and when it has enough evidence.
  This path can follow cross-references and use conversation state, at variable
  model cost.

Both run through the shared `flows._runner` observability wrapper so tracing,
turn limits, and the memory hook behave as they do for every other agent. The
navigation primitive takes a tree, a question, and an optional set of seed node
ids (see [Hybrid Search Compatibility](#hybrid-search-compatibility)); it does
not own persistence, embedding, or answer generation.

## Multi-Model Compatibility

The design must not hard-lock to one provider for either generation or
embeddings. On `master` today `cfg.model` is a plain string passed straight to
`agents.Agent(model=...)`, there is no provider-detection helper, and the only
embedding provider is OpenAI. The design commits to two seams.

- **One model-resolution seam for generation.** Structuring, single-pass
  classification, and agentic traversal all take their model from
  `BaseFlowCfg.model`. Provider resolution is centralized in one helper that maps
  a model string to an SDK model, using the Agents SDK LiteLLM integration
  (`agents.extensions.models.litellm_model.LitellmModel`, and the
  `litellm/<provider>/<model>` convention) for non-OpenAI providers. `litellm`
  is already a pinned dependency. No call site constructs a provider inline and
  no per-flow model registry is added.
- **Provider-agnostic embeddings for the hybrid pre-filter.** The library
  already isolates embeddings behind the private `_EmbeddingProvider` Protocol
  (`embed(texts, *, model, dimensions)` / `close()`). Hybrid navigation depends
  only on that seam and on `SemanticQuery`/`SemanticHit`, never on a specific
  embedding vendor. Promoting the seam to a supported swap point is a library
  decision tracked separately; navigation must not assume OpenAI.

Multi-model support is therefore a property of these two seams, not a provider
registry or a retriever hierarchy — both explicit non-goals.

## Hybrid Search Compatibility

Hybrid retrieval — use semantic search to shortlist nodes, then let the agent
reason over that shortlist — is a first-class future mode, and the current
storage model already makes it nearly free.

- When a navigation `TreeKnowledge` is stored, `library` produces one projection
  for every non-root node. `search(SemanticQuery(...))` over those projections
  returns `SemanticHit` values whose `node_id` identifies candidate nodes in the
  same tree.
- The navigation primitive accepts optional seed node ids. Hybrid mode is the
  composition: run `library.search()`, map the hits' `node_id` values to seeds,
  and start agentic traversal from those nodes instead of the root; the agent
  still expands, drills down, and follows cross-references from there.
- Embeddings stay a coarse pre-filter. The agent may leave the seeded subtree,
  and a hit never becomes an answer without the reasoning step. This is the
  `TreeKnowledge` contract restated, not a new one.
- The candidate source is pluggable. Pure-agentic navigation passes no seeds;
  hybrid passes library seeds; a future corpus-level policy could pass seeds from
  a different index without changing the navigation signature.

Because the seed interface is just node ids, the pure-agentic MVP and the hybrid
mode share one primitive; hybrid adds a shortlist step in front of it rather than
a second retrieval path.

## Boundaries and Import Contracts

Placement must satisfy the seven `import-linter` contracts. Relevant directions:
`knowledge` is a leaf; `library` imports only `knowledge`; `rag` imports only
`preprocess`; `flows` + `magic` is the apex.

- `quantmind.mind` does not exist yet and is only named as a forbidden import
  inside the `library` and `rag` contracts. When implemented, add a contract
  pinning `mind -> library -> knowledge`: `mind` may import `knowledge`,
  `library`, `configs`, and `utils`, but not `flows`, `magic`, `rag`, or
  `preprocess`. `library` and `rag` remain barred from importing `mind`.
- The stateless draft helper stays in `rag` and keeps `rag`'s
  imports-only-`preprocess` rule intact.
- Canonical tree construction stays in `flows`, which already sits above
  `knowledge` and `library`.

The existing `## PageIndex Boundary` (library) and `## Collection Search and
PageIndex` (rag) sections should cross-link this page and note that agentic,
library-backed navigation is owned by `mind`, while `rag` remains draft-only.

## Phasing

Each phase is independently shippable and testable.

- **P0 — Build and persist.** Deterministic outline signals in `preprocess`, the
  `Citation.end_page` addition, the `flows` structuring step producing a
  verified canonical `TreeKnowledge` linked to a `PaperSourceRevision`, and
  `library.put()` persistence with per-node projections.
- **P1 — Agentic navigation.** The `mind.navigation` primitive with single-pass
  classification and agentic traversal over the persisted tree, returning node
  evidence. Uses the single model-resolution seam.
- **P2 — Hybrid pre-filter.** Seed navigation from `library.search()` node hits.
  No new storage; a shortlist step in front of P1.
- **P3 — Corpus navigation.** Extend seeds and traversal across many documents.
  Design-only here; it must not add a provider registry or a second store.

## Relationship to Prior Art

`VectifyAI/PageIndex` (MIT) validates the approach: table-of-contents detection,
recursive node subdivision, a node schema of title / id / page-range / summary /
children, and two retrieval paths (single-pass node selection and an Agents-SDK
tool loop). QuantMind reuses these mechanics where useful but keeps its own
tree, financial-time, provenance, and citation contracts, and treats PDF content
as untrusted input to structuring calls. Reuse of PageIndex code remains subject
to license, dependency, and contract review; commodity mechanics are not
reimplemented before that review.

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
