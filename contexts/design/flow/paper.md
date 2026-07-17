# Paper Extraction: End-to-End Design

## Quick Summary

- **Purpose**: Define how a paper input becomes a validated `Paper`.
- **Read when**: Changing paper inputs, parsing, section trees, source tracking, page ranges, or future PageIndex support.
- **Status**: Planned design; [Current Gaps](#current-gaps) lists what is not implemented yet.
- **Core rule**: A model or PageIndex may suggest a section tree. Code creates the final IDs, links, order, page ranges, citations, and source-backed text.
- **Page numbering**: PDF page ranges start at 1 and include both the first and last page.

## Contents

- [Overview](#overview)
- [Who Owns Each Step](#who-owns-each-step)
- [Supported Inputs and Fetching](#supported-inputs-and-fetching)
- [Keep PDF Pages Separate](#keep-pdf-pages-separate)
- [Which Source Provides Each Field](#which-source-provides-each-field)
- [Model Output Is a Draft](#model-output-is-a-draft)
- [Build and Validate the Final Paper](#build-and-validate-the-final-paper)
- [Rules for a Valid Paper](#rules-for-a-valid-paper)
- [Get Node Content from Source Pages](#get-node-content-from-source-pages)
- [Retries and Failures by Step](#retries-and-failures-by-step)
- [What Paper Extraction Does Not Do](#what-paper-extraction-does-not-do)
- [How PageIndex Can Fit Later](#how-pageindex-can-fit-later)
- [Fixed Paper Test Data](#fixed-paper-test-data)
- [Current Gaps](#current-gaps)

## Overview

Paper extraction has six steps and returns one validated result:

```mermaid
flowchart LR
    input["Paper input"] --> resolve["Resolve and fetch"]
    resolve --> source["Keep pages separate<br/>and record source facts"]
    source --> draft["Suggest a section tree"]
    draft --> assembly["Build the final Paper in code"]
    assembly --> validate["Validate the tree and source links"]
    validate --> paper["Validated Paper"]
```

Code, not the model, records source facts, creates IDs and tree links, copies
text from source pages, creates citations, and validates the result. A model or
future PageIndex integration may suggest sections, titles, summaries, and page
ranges, but that output remains a draft.

Paper extraction does not require PageIndex. It first preserves ordered source
pages and passes them through a small `PaperSourceDocument -> draft` interface.
PageIndex can later implement that interface without changing the `Paper` type
or creating the final IDs.

## Who Owns Each Step

This operation creates one extraction result. It does not store papers, search
across them, or write answers.

| Work | Owner |
|---|---|
| Resolve identifiers, fetch bytes, parse pages, and hash source content | `quantmind.preprocess` |
| Configure the operation and select the input variant | `quantmind.configs` |
| Suggest a section tree and build the final `Paper` | `quantmind.flows` |
| Define the `Paper`, `TreeKnowledge`, `TreeNode`, source, citation, and extraction models | `quantmind.knowledge` |
| Store knowledge and make it searchable by meaning | `quantmind.library` |
| Navigate a tree and write an answer | A future `quantmind.mind` consumer or agent application |

The `flow/` directory groups this work because it spans several packages. It
does not require a new `*_flow` public API and does not override the
[public operation naming rules](../operations/naming.md).

## Supported Inputs and Fetching

### Supported inputs

The planned pipeline accepts the existing `PaperInput` types with the following
behavior:

| Input | What happens |
|---|---|
| `ArxivIdentifier` | Resolve the identifier to an exact arXiv version, provider metadata, and PDF bytes. Preserve the resolved URL and the time that version became available. |
| `HttpUrl` | Follow a limited number of redirects and accept supported PDF, HTML, Markdown, or plain-text content. Record the final URL and exact fetched content. |
| `LocalFilePath` | Read a supported PDF, HTML, Markdown, or plain-text file. The caller owns file retention; extraction records a local source reference and content hash. |
| `RawText` | Treat the supplied text as one source document without pages. It can produce a `Paper`, but it cannot claim PDF page ranges. |

`DoiIdentifier` remains unsupported until an open-access resolver can fetch the
exact paper content. A DOI landing page alone contains metadata and may not
contain the paper itself.

### Explicitly unsupported inputs

V1 does not claim support for:

- password-protected or corrupt PDFs;
- image-only scans that require OCR;
- authenticated, paywalled, or dynamically rendered sources without a
  separately authorized fetcher;
- unsupported binary or media content types;
- a DOI that cannot be resolved to accessible paper content;
- a source whose exact fetched content cannot be identified.

Unsupported input fails before a model or tree builder runs. The pipeline must
not send an error page, login page, or unresolved metadata page to a model and
call the result a successfully extracted paper.

### Fetching rules

Fetching produces one exact source version. Resolve redirects, arXiv revisions,
and the content selected by the server before parsing. The final URI and
SHA-256 hash identify the exact bytes used. A retry may receive changed
content; when the hash changes, treat it as a new source version rather than
silently merging it with the earlier attempt.

## Keep PDF Pages Separate

PDF parsing must preserve ordered pages before any tree builder runs. The next
step receives data shaped like this:

```text
PaperSourceDocument
├── source details and exact content hash
├── page or character range format
└── ordered pages
    ├── page number
    ├── extracted text, including an empty string when the page has no text
    └── optional parser-provided layout or outline hints
```

Required properties:

- PDF page ranges in the fixed test PDF, model draft, and citations start at 1
  and include both the first and last page.
- Every physical page remains represented and in order. Empty pages are not
  dropped because doing so would renumber later page references.
- Text cleanup may remove parser noise, but it must not erase the page
  boundary or change which page owns an anchor.
- Record the exact source hash, parser name and version, and cleanup version
  with the extraction run.
- HTML, Markdown, plain text, and `RawText` have no pages. They retain source
  text and character positions but do not invent PDF page numbers.
- A page-based tree builder, including a future PageIndex integration, accepts
  only a source document whose range unit is `pdf_page`.

`PaperSourceDocument` is a temporary value used during extraction, not another
public knowledge model. The caller keeps raw PDF or HTML files; this design
does not embed them inside `Paper`.

## Which Source Provides Each Field

Facts from the fetched source take priority over model suggestions. A model may
fill a missing summary or section title, but it must not overwrite a URL,
content hash, publication time, or other known source fact.

| Field or fact | Who sets it and where it is stored |
|---|---|
| Resolved source URI and source kind | Resolver/fetcher; `SourceRef.kind` and `SourceRef.uri` |
| Exact fetched bytes and content hash | Fetcher; `SourceRef.content_hash`, while the caller keeps the raw file |
| Fetch time | Fetcher clock; `SourceRef.fetched_at` |
| Publication/version time | Provider metadata for the exact version; do not replace it with the first-version date |
| Source availability time | Provider metadata for the exact version; `Paper.available_at`. Use fetch time as the latest possible availability time only when the publication time is unknown. |
| Latest date covered by the paper | A study or data cutoff stated by the source; otherwise use the exact version's publication time for `Paper.as_of`, never fetch time |
| Authors and title | Provider or document metadata; `Paper.authors` and root title unless that metadata is missing or clearly invalid |
| Extraction model, operation, run, and time | Runtime; `ExtractionRef` |
| Parser and cleanup versions | Runtime; keep them with the extraction run until `Paper` has a dedicated field |
| Summaries, section titles, methodology, findings, and limitations | Model or tree builder; validate them before building the final `Paper` |
| Final item/node IDs and tree links | Code only |

For an arXiv revision, `available_at` refers to the exact revision used, not the
first submission date. `as_of` is the latest date covered by the paper, not the
time it was fetched. Unknown publication or availability data remains unknown
rather than being guessed by a model.

## Model Output Is a Draft

A model or PageIndex integration returns a draft with only the information
needed to suggest a section tree. It may contain:

- temporary node keys used only by that integration;
- candidate titles and summaries;
- an ordered parent/child outline using those temporary keys;
- suggested page ranges using the source document's range format;
- confidence values or errors needed to accept, repair, or reject the draft.

A draft must not set the final `Paper.id`, `TreeNode.node_id`,
`parent_id`/`children_ids`, copied source text, known source facts, or a
object tied to one provider and exposed through the public result.

The interface is intentionally small: one `PaperSourceDocument` produces one
draft. The default implementation can use an LLM; a future PageIndex
integration can return the same draft shape. Neither defines the public
`Paper` type.

## Build and Validate the Final Paper

Code treats the accepted draft as untrusted input and performs these steps:

1. Generate the final item and node IDs. Temporary integration IDs never
   appear in the returned result.
2. Set `parent_id` and ordered `children_ids` together for each parent-child
   link so they cannot disagree.
3. Assign sibling positions from the accepted order.
4. Check page ranges against the preserved source pages.
5. Copy source content from the stated page range and build citations from the
   same pages. Model-written paraphrases never become source text or quotes.
6. Apply known source facts and extraction run details.
7. Construct the final `Paper` and run every validation rule below.

Code owns the final IDs, but separate extraction runs do not need to create the
same IDs. Merging results across runs is a separate decision.

## Rules for a Valid Paper

A successful `Paper` satisfies all of the following before it is returned:

### IDs and root

- `root_node_id` identifies exactly one entry in `nodes`.
- Every dictionary key equals that node's `node_id`.
- The root has `parent_id=None`; every other node has exactly one parent.
- All item and node IDs are created by code and unique within the result.

### Parent and child links

- Every referenced parent and child exists.
- A parent lists each child, and each child points back to that parent.
- A child appears at most once in a parent's `children_ids`.
- The same accepted draft produces the same sibling order, and sibling
  positions are unique within one parent.

### Reachability and safe traversal

- Every node is reachable from the root.
- The graph has no cycles and no node points to itself.
- No node is shared by multiple parents.
- Depth-first walk and root-to-node path lookup finish for every node. Looking
  up an unknown node returns the documented safe result.

### Source page ranges and citations

- A PDF page range uses `pdf_page`, starts at 1 or later, has
  `start_page <= end_page`, and ends within the preserved PDF page count.
- Citation page ranges also start at 1, include both ends, and stay within the
  PDF page count.
- Citation quotes and node content come from the identified source range.
- Inputs without pages do not carry made-up PDF page ranges.
- Sibling page ranges may overlap. A child range is not required to fit
  completely inside its parent range unless a future rule adds that
  requirement.

## Get Node Content from Source Pages

Branch nodes may retain source content. `content=None` remains useful for a
navigation-only node, but being a branch is not itself a reason to discard its
source text.

When content is present, code copies it from the node's 1-based inclusive page
range. Page-level ranges can include a heading or paragraph also used by an
adjacent node; this is expected when ranges overlap. Do not ask the model to
reproduce source text or create a parent node by joining model summaries.

Choosing a section and loading its source text remain separate operations. An
agent can read titles and summaries to select a node, then fetch that node's
page range. A future PageIndex integration may help build the outline, but it
does not fetch the source text.

## Retries and Failures by Step

| Step | Same input, same result? | Retry and failure behavior |
|---|---|---|
| Input validation and fetching rules | Yes, for one configuration | Invalid or unsupported input fails immediately. |
| Network fetch | No; the remote source may change | Retry only a limited number of temporary failures. Record and hash the exact successful response. Permanent HTTP or content failures stop the operation. |
| PDF or text parsing | Yes, for the same bytes and parser version | Parser failure stops before the model or tree builder runs. Do not skip corrupt pages or renumber later pages. |
| Model or PageIndex draft | No when a model is used | Limited retries may repair network or invalid-draft failures against the same source version. Record every attempt. |
| Build the final `Paper` | Yes for one accepted draft and source version, except for new UUID values | Invalid links, page ranges, or conflicts with source facts reject the draft. |
| Final validation | Yes | Any failure rejects the entire result; never return a partial `Paper` as success. |

A successful result means one `Paper` passed its source, page-range, and tree
checks. Fetching bytes, receiving a model response, or constructing a Pydantic
object alone is not success. A failure records enough about the failed step and
source to diagnose or retry it, but it does not store a partial `Paper`.

## What Paper Extraction Does Not Do

Paper extraction returns one validated `Paper`. Other components handle:

- storing and safely updating it in `LocalKnowledgeLibrary`;
- generating embeddings and building search records;
- searching across a collection;
- PageIndex-style tree navigation within one selected document;
- writing answers, managing conversation state, and citing a final response;
- deciding whether to keep raw source bytes.

Keeping this work separate allows Paper extraction to be implemented before
PageIndex. PageIndex can then be added without replacing collection-wide search
or stored knowledge.

## How PageIndex Can Fit Later

Future integration must preserve these decisions:

1. PageIndex receives ordered source pages before they are joined into one text.
2. Its node IDs remain temporary. Code creates the final IDs and tree links.
3. It may propose titles, summaries, ordering, and 1-based inclusive page
   ranges through the draft described above.
4. It does not become the `Paper`, `TreeKnowledge`, or `TreeNode` type.
5. Navigation uses titles and summaries; loading source text separately fetches
   the selected page range.
6. Sibling page ranges may overlap, and a child range does not need to fit
   completely inside its parent range.

## Fixed Paper Test Data

The fixed test files live at:

```text
tests/fixtures/paper/golden/
├── paper.pdf
└── expected.json
```

It is a small four-page test paper with a nested subsection, multi-page
sections, and intentionally overlapping sibling page ranges. `expected.json`
contains only stable facts: page count, titles and paths, 1-based inclusive
page ranges, distinctive text anchors, tree shape, and validation rules. It
does not pin summaries or any other wording produced by a model.

The offline test parses the fixed PDF, checks every text anchor against its
declared page, validates the tree and all rules, and confirms that overlapping
ranges work. Paper extraction and future PageIndex work must reuse these files
rather than create a competing test paper.

## Current Gaps

The repository does not yet guarantee the target pipeline above:

- `pdf_to_markdown()` concatenates non-empty page text and drops page
  boundaries and empty pages.
- `paper_flow()` sends the flattened document to one extraction agent and asks
  it to return the final `Paper` directly.
- The model currently controls IDs, edges, citations, source fields, and
  content instead of returning the limited draft described above.
- Resolved source URL, content hash, publication/version time, fetch time, and
  exact-version availability are not consistently collected from the source.
- `DoiIdentifier` raises `NotImplementedError` because there is no accessible
  content resolver.
- `TreeKnowledge` provides traversal helpers but does not yet check every root,
  link, reachability, cycle, or page-range rule when it is created.
- The structured-output failure tracked by issue #91 remains separate from
  this design issue.

Those gaps are future implementation work. Adding PageIndex first would not fix
them and is not required to implement the Paper steps above.
