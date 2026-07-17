# Parse PDFs without losing page or visual context

## Quick Summary

- **Purpose**: Define the deterministic PDF value shared by paper extraction, collection-wide RAG, and a future PageIndex adapter.
- **Read when**: Changing PDF parsing, page artifacts, LlamaIndex ingestion, or multimodal evidence.
- **Status**: Implemented by `quantmind.preprocess.parse_pdf` and the private LlamaIndex conversion helpers.
- **Core rule**: Parsing preserves every physical page and its source coordinates before any chunker or tree builder runs.

## Contents

- [Parsed document boundary](#parsed-document-boundary)
- [Artifacts and ownership](#artifacts-and-ownership)
- [LlamaIndex conversion](#llamaindex-conversion)
- [PageIndex compatibility](#pageindex-compatibility)

## Parsed document boundary

`parse_pdf()` returns a frozen QuantMind `ParsedDocument`. It records the SHA-256 hash of the exact PDF bytes, parser and cleanup versions, and one ordered `ParsedPage` for every physical page. Page numbers are 1-based. An empty page remains present with empty text and no blocks.

Each `TextBlock` keeps its page ownership, text, bounding box, and parser-provided font and confidence values when available. Bounding boxes use PDF page coordinates `(x0, y0, x1, y1)`. These deterministic preprocessing values are not canonical knowledge models.

## Artifacts and ownership

The caller chooses an artifact directory. When supplied, parsing renders one PNG screenshot per page and stores a stable path reference on the page. The library does not copy screenshots or PDF bytes into canonical SQLite knowledge. Without an artifact directory, pages remain valid and `screenshot_path` is absent.

## LlamaIndex conversion

LlamaIndex is the required RAG data plane. QuantMind converts pages to private LlamaIndex documents, retaining source hash, page number, block coordinates, and screenshot references as metadata. `chunk_parsed_document()` applies LlamaIndex `SentenceSplitter`; supported splitter arguments pass through `SentenceSplitterConfig`.

`retrieve_parsed_document()` performs a bounded BM25 retrieval over those chunks and converts results back to `ParsedDocumentHit`. Public QuantMind values never expose LlamaIndex `Document`, node, index, or retriever types.

Flattened Markdown remains a compatibility view produced from the preserved pages. It is not the primary parsing result.

## PageIndex compatibility

PageIndex may later consume the same ordered `ParsedDocument` to propose a document tree and navigate within a selected long document. It remains independent of collection-wide LlamaIndex ranking and is not forced through `LocalKnowledgeLibrary.search()`.
