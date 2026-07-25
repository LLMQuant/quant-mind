# Parse PDFs without losing page or visual context

## Quick Summary

- **Purpose**: Define the deterministic PDF value shared by paper extraction, document RAG, and structure-tree outline signals.
- **Read when**: Changing PDF parsing, page artifacts, or multimodal source evidence.
- **Status**: Implemented by `quantmind.preprocess.format.parse_pdf`.
- **Core rule**: Parsing preserves every physical page and its source coordinates before any chunker or semantic artifact producer runs.

## Contents

- [Parsed document boundary](#parsed-document-boundary)
- [Artifacts and ownership](#artifacts-and-ownership)
- [Downstream consumers](#downstream-consumers)

## Parsed document boundary

`parse_pdf()` returns a frozen QuantMind `ParsedDocument`. It records the SHA-256 hash of the exact PDF bytes, parser and cleanup versions, and one ordered `ParsedPage` for every physical page. Page numbers are 1-based. An empty page remains present with empty text and no blocks.

Each `TextBlock` keeps its page ownership, text, bounding box, and parser-provided font and confidence values when available. Bounding boxes use PDF page coordinates `(x0, y0, x1, y1)`. These deterministic preprocessing values are not canonical knowledge models.

## Artifacts and ownership

The caller chooses an artifact directory. When supplied, parsing renders one PNG screenshot per page and stores a path reference on the page. Without an artifact directory, pages remain valid and `screenshot_path` is absent.

`ParsedDocument` paths and bytes are preprocessing values, not canonical storage. A source-first paper flow reads the referenced bytes, converts them to content-addressed `PaperAssetRef` values, and carries the blobs in `PaperSourceRevision` until `LocalKnowledgeLibrary.put_paper()` copies them into linked SQLite rows. Other parser callers continue to own their files.

## Downstream consumers

Preprocessing ends after producing `ParsedDocument`. It does not chunk, index, rank, or answer a query.

- [`quantmind.rag`](../rag/document.md) converts the parsed value into LlamaIndex-backed chunks and page-aware retrieval evidence.
- The [paper flow](../flow/paper.md) (`PaperFlow(PaperSemanticCfg).build`) uses the preserved source pages to build an exact source revision and a canonical chunk set before generating a cited summary.
- `extract_outline_signals()` consumes the same ordered pages to emit deterministic table-of-contents, heading, and printed-page-offset hints for structure-tree construction.

Flattened Markdown remains a compatibility view produced from the preserved pages. It is not the primary parsing result.
