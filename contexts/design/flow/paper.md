# Paper End-to-End Design Contract

- **Status**: Target contract with current gaps called out below
- **Scope**: Paper source resolution through canonical `Paper` assembly
- **Golden span convention**: PDF pages, 1-based and inclusive

## Decision Summary

Paper extraction is a staged operation with one canonical output boundary:

```text
Paper input
  -> resolve and fetch
  -> page-preserving source document plus authoritative metadata
  -> structural extraction draft
  -> deterministic canonical assembly
  -> tree and provenance validation
  -> Paper
```

Deterministic code owns source evidence, canonical identity, graph structure,
page slicing, citations, and validation. A model or future PageIndex adapter
may propose semantic structure, titles, summaries, and source spans, but its
output is a draft rather than a canonical `Paper`.

PageIndex is not a prerequisite. The pipeline first preserves ordered source
pages and uses a narrow tree-builder seam; PageIndex can later implement that
seam without changing the `Paper` schema or taking ownership of canonical IDs.

## Product and Ownership Boundary

This contract covers one extraction result. It does not make Paper extraction
responsible for persistence, retrieval, or answer generation.

| Concern | Owner |
|---|---|
| Resolve identifiers, fetch bytes, parse pages, and hash source content | `quantmind.preprocess` |
| Configure the operation and select the input variant | `quantmind.configs` |
| Produce semantic structure and assemble a canonical result | `quantmind.flows` |
| Define `Paper`, `TreeKnowledge`, `TreeNode`, source, citation, and extraction schemas | `quantmind.knowledge` |
| Persist and semantically index canonical knowledge | `quantmind.library` |
| Navigate a tree and synthesize an answer | A future `quantmind.mind` consumer or agent application |

`flow/` is a documentation grouping for this cross-domain behavior. It does
not require a new `*_flow` public API and does not override the
[public operation naming contract](../operations/naming.md).

## Inputs and Source Resolution

### Supported input intents

The target pipeline accepts the existing `PaperInput` intents with the
following semantics:

| Input | Resolution behavior |
|---|---|
| `ArxivIdentifier` | Resolve the identifier to an exact arXiv version, authoritative metadata, and PDF bytes. Preserve the resolved source URL and version-specific availability time. |
| `HttpUrl` | Follow bounded redirects and accept supported PDF, HTML, Markdown, or plain-text content. Record the final canonical URL and fetched representation. |
| `LocalFilePath` | Read a supported PDF, HTML, Markdown, or plain-text file. The caller owns file retention; extraction records a local source reference and content hash. |
| `RawText` | Treat the supplied text as one non-paginated source document. It can produce a `Paper`, but it cannot claim PDF page spans. |

`DoiIdentifier` remains explicitly unsupported until an open-access resolver
can produce an exact fetchable representation. A DOI landing page alone is
metadata, not necessarily the paper content.

### Explicitly unsupported inputs

V1 does not claim support for:

- password-protected or corrupt PDFs;
- image-only scans that require OCR;
- authenticated, paywalled, or dynamically rendered sources without a
  separately authorized fetch adapter;
- unsupported binary or media content types;
- a DOI that cannot be resolved to accessible paper content;
- a source whose exact fetched representation cannot be identified.

Unsupported input fails before semantic extraction. The pipeline must not send
an error page, login page, or unresolved metadata page to a model and call the
result a successfully extracted paper.

### Resolution rules

Resolution produces one exact source version. Redirects, arXiv revisions, and
content negotiation are resolved before parsing. The resolved URI and SHA-256
hash identify the bytes used for this extraction. A retry may fetch a changed
representation; if its content hash changes, it is a new source version and
must not be merged silently with an earlier attempt.

## Page-Preserving Source Document

PDF parsing must preserve ordered pages before any tree builder runs. The
conceptual deterministic handoff has this shape:

```text
PaperSourceDocument
├── source metadata and exact content hash
├── span unit and indexing convention
└── ordered pages
    ├── page number
    ├── extracted text, including an empty string when the page has no text
    └── optional deterministic layout or outline signals
```

Required properties:

- PDF page numbers are 1-based and inclusive everywhere in the Paper golden
  contract, structural drafts, and citations.
- Every physical page remains represented and in order. Empty pages are not
  dropped because doing so would renumber later evidence.
- Text normalization may remove parser noise, but it must not erase the page
  boundary or change which page owns an anchor.
- The exact source hash, parser identity/version, and any deterministic
  normalization identity are available to provenance assembly.
- HTML, Markdown, plain text, and `RawText` are non-paginated. They retain
  source text and character evidence but do not invent PDF page numbers.
- A page-based tree builder, including a future PageIndex adapter, accepts only
  a source document whose span unit is `pdf_page`.

The page-preserving document is an extraction intermediate, not a second
canonical knowledge schema. Raw PDF/HTML retention remains caller-owned and is
not embedded inside `Paper` by this contract.

## Authoritative Metadata

Source-derived metadata and model-derived semantics have different authority.
The model may fill an explicitly missing semantic field, but it may not
overwrite authoritative source evidence.

| Field or fact | Authority and canonical destination |
|---|---|
| Resolved source URI and source kind | Resolver/fetcher; `SourceRef.kind` and `SourceRef.uri` |
| Exact fetched bytes and content hash | Fetcher; `SourceRef.content_hash` plus caller-owned raw artifact |
| Fetch time | Fetcher clock; `SourceRef.fetched_at` |
| Publication/version time | Authoritative provider metadata; establishes when the exact version became public and must not be replaced with the first-version date |
| Source availability time | Exact-version provider metadata when available; `Paper.available_at`. Fetch time is a conservative upper bound when publication availability is unknown. |
| Information cutoff | A source-declared study/data cutoff when present; otherwise the exact version publication time is a documented conservative fallback for `Paper.as_of`, never fetch time |
| Authors and authoritative title | Provider or document metadata; `Paper.authors` and root title unless the metadata is missing or demonstrably invalid |
| Extraction model, operation, run, and time | Runtime; `ExtractionRef` |
| Parser and normalization identity | Deterministic runtime provenance; retained with the extraction run until the canonical item schema has an explicit item-level field |
| Summaries, semantic section titles, methodology, findings, and limitations | Structural draft producer; validated before canonical assembly |
| Canonical item/node IDs and graph links | Deterministic assembly code only |

For an arXiv revision, `available_at` refers to the exact revision used, not the
first submission date. `as_of` is the information cutoff represented by the
paper; it must remain distinct from fetch time. Unknown publication or
availability metadata remains unknown rather than being guessed by a model.

## Structural Extraction Draft

A draft is the only provider- or model-facing structural contract. It may
contain:

- adapter-local node keys used only within the draft;
- candidate titles and summaries;
- an ordered parent/child outline expressed with draft-local references;
- candidate page spans using the source document's explicit span unit;
- confidence or diagnostic information needed to accept, repair, or reject the
  draft.

A draft must not contain canonical `Paper.id`, `TreeNode.node_id`, canonical
`parent_id`/`children_ids`, copied source text, authoritative metadata, or a
provider-specific object that leaks through the public result.

The structural producer is intentionally narrow: conceptually it maps one
`PaperSourceDocument` to one structural draft. The default implementation can
use an LLM; a future PageIndex adapter can provide the same kind of draft.
Neither becomes the canonical `Paper` schema.

## Deterministic Canonical Assembly

Assembly treats the accepted draft as untrusted input and performs these steps
in code:

1. Generate canonical item and node IDs. External or draft-local IDs never
   escape their adapter.
2. Resolve each draft parent reference and derive both `parent_id` and ordered
   `children_ids` from one checked relation.
3. Assign sibling positions deterministically from the accepted order.
4. Validate and normalize page spans against the preserved source pages.
5. Slice source content from the inclusive page range and construct citations
   from the same source evidence. Model-generated paraphrases never become
   source content or quotes.
6. Apply authoritative metadata and runtime extraction provenance.
7. Construct the canonical `Paper` and run the complete invariant validator.

Canonical IDs are code-owned, but this contract does not require them to be
identical across independent extraction runs. Stable cross-run identity is a
separate deduplication decision.

## Tree and Paper Invariants

A successful `Paper` satisfies all of the following before it is returned:

### Identity and root

- `root_node_id` identifies exactly one entry in `nodes`.
- Every dictionary key equals that node's `node_id`.
- The root has `parent_id=None`; every other node has exactly one parent.
- All canonical item and node IDs are code-owned and unique within the result.

### Edges and ordering

- Every referenced parent and child exists.
- Parent/child relationships are bidirectionally consistent.
- A child appears at most once in a parent's `children_ids`.
- Sibling order is deterministic, and sibling positions are unique within one
  parent.

### Reachability and safe traversal

- Every node is reachable from the root.
- The graph is acyclic and contains no self-edge.
- No node is shared by multiple parents.
- Depth-first walk and root-to-node path lookup terminate for every canonical
  node. Unknown node lookup returns the documented safe result rather than
  following unchecked links.

### Source spans and citations

- A PDF span uses `pdf_page`, starts at 1 or later, has
  `start_page <= end_page`, and ends within the preserved PDF page count.
- Citation page ranges obey the same 1-based inclusive convention and source
  bounds.
- Citation quotes and node content come from the identified source range.
- Non-paginated inputs do not carry fabricated PDF spans.
- Sibling spans may overlap. A child span is not required to be strictly
  contained by its parent's span unless a later canonical rule adds that
  independent requirement.

## Branch Content and Source Slicing

Branch nodes may retain source content. `content=None` remains useful for a
navigation-only node, but being a branch is not itself a reason to discard its
evidence.

When content is present, deterministic code slices it from the node's tight
1-based inclusive source page range. Page-granular ranges can include a heading
or paragraph also used by an adjacent node; this is expected when spans
overlap. Assembly must not ask the model to reproduce source text, and it must
not derive a parent node's content by concatenating model summaries.

Navigation and evidence loading remain separate operations: an agent can read
titles and summaries to select a node, then fetch the selected node's tight
source page range. A future PageIndex adapter may help build the outline, but
it does not own source-range fetching.

## Determinism, Retry, and Failure

| Stage | Determinism | Retry and failure contract |
|---|---|---|
| Input validation and resolution policy | Deterministic for one configuration | Invalid or unsupported input fails immediately. |
| Network fetch | Externally variable | Retry only bounded transient failures. Record and hash the exact successful response. Permanent status/content failures stop the operation. |
| PDF or text parsing | Deterministic for fixed bytes and parser version | Parser failure stops before semantic extraction. Do not skip corrupt pages or renumber around them. |
| Structural draft production | Nondeterministic when model-backed | Bounded retries may repair transport or draft-schema failures against the same source version. Each attempt is observable. |
| Canonical assembly | Deterministic for one accepted draft and source version, except generated UUID values | Invalid references, spans, or authoritative metadata conflicts reject the draft. |
| Invariant validation | Deterministic | Any failure rejects the entire result; no partial `Paper` is returned as success. |

A successful result means one canonical `Paper` passed provenance, span, and
tree validation. Fetching bytes, obtaining a model response, or constructing a
Pydantic object alone is not success. Failure preserves enough stage and source
identity to diagnose or retry the operation, but it does not persist partial
canonical knowledge implicitly.

## Output Boundary and Downstream Consumers

Paper extraction returns canonical knowledge. The following are explicit
downstream consumers and stay outside this operation:

- persistence and idempotent storage in `LocalKnowledgeLibrary`;
- embedding generation and semantic indexing;
- semantic retrieval across a collection;
- PageIndex-style tree navigation within one selected document;
- answer synthesis, conversational state, and citations in a final response;
- caller policy for retaining raw source bytes.

Keeping these concerns separate lets Paper extraction land before PageIndex and
lets PageIndex arrive later without replacing semantic retrieval or the
canonical library.

## Future PageIndex Compatibility Seam

Future integration must preserve these decisions:

1. PageIndex receives ordered source pages before any flattening step.
2. Its node IDs remain adapter-local. Canonical IDs and graph links are created
   during deterministic assembly.
3. It may propose titles, summaries, ordering, and 1-based inclusive page
   spans through the structural draft seam.
4. It does not become the `Paper`, `TreeKnowledge`, or `TreeNode` schema.
5. Outline navigation uses titles and summaries; evidence loading separately
   fetches the selected tight source range.
6. Integration does not assume non-overlapping sibling spans or strict child
   containment.

## Golden Fixture Contract

The repository-owned fixture lives at:

```text
tests/fixtures/paper/golden/
├── paper.pdf
└── expected.json
```

It is a small four-page synthetic paper with a nested subsection, multi-page
sections, and intentionally overlapping sibling page spans. `expected.json`
contains only stable facts: page count, titles and paths, 1-based inclusive
spans, distinctive text anchors, topology, and named invariants. It does not
pin summaries or any other wording produced by a model.

The offline contract test parses the fixed PDF, checks every anchor against its
declared page, validates topology and invariants, and confirms the overlap case
remains representable. Paper extraction and future PageIndex work must reuse
this fixture rather than create a competing golden document.

## Current Behavior and Known Gaps

The repository does not yet guarantee the target pipeline above:

- `pdf_to_markdown()` concatenates non-empty page text and drops page
  boundaries and empty pages.
- `paper_flow()` sends the flattened document to one extraction agent and asks
  it to return the canonical `Paper` directly.
- The model currently controls IDs, edges, citations, source fields, and
  content instead of returning a restricted structural draft.
- Resolved source URL, content hash, publication/version time, fetch time, and
  exact-version availability are not assembled authoritatively end to end.
- `DoiIdentifier` raises `NotImplementedError` because there is no accessible
  content resolver.
- `TreeKnowledge` provides traversal helpers but does not yet enforce the full
  root, edge, reachability, acyclicity, or span invariant set at construction.
- The structured-output failure tracked by issue #91 remains separate from
  this design issue.

Those gaps are implementation work after this contract. Adding PageIndex first
would not fix them and is not required to implement the staged Paper pipeline.
