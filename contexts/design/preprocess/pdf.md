# Parse PDFs without losing page or visual context

## Quick Summary

- **Purpose**: Define the deterministic PDF value shared by paper extraction, document RAG, and a future PageIndex adapter.
- **Read when**: Changing PDF parsing, page artifacts, or multimodal source evidence.
- **Status**: Implemented by `quantmind.preprocess.format.parse_pdf`.
- **Core rule**: Parsing preserves every physical page and its source coordinates before any chunker or tree builder runs.

## Contents

- [Parsed document boundary](#parsed-document-boundary)
- [Artifacts and ownership](#artifacts-and-ownership)
- [Downstream consumers](#downstream-consumers)

## Parsed document boundary

`parse_pdf()` returns a frozen QuantMind `ParsedDocument`. It records the SHA-256 hash of the exact PDF bytes, parser and cleanup versions, and one ordered `ParsedPage` for every physical page. Page numbers are 1-based. An empty page remains present with empty text and no blocks.

Each `TextBlock` keeps its page ownership, text, bounding box, and parser-provided font and confidence values when available. Bounding boxes use PDF page coordinates `(x0, y0, x1, y1)`. These deterministic preprocessing values are not canonical knowledge models.

## Artifacts and ownership

The caller chooses an artifact directory. When supplied, parsing renders one PNG screenshot per page and stores a stable path reference on the page. The library does not copy screenshots or PDF bytes into canonical SQLite knowledge. Without an artifact directory, pages remain valid and `screenshot_path` is absent.

## Downstream consumers

Preprocessing ends after producing `ParsedDocument`. It does not chunk, index, rank, or answer a query.

- [`quantmind.rag`](../rag/document.md) converts the parsed value into LlamaIndex-backed chunks and page-aware retrieval evidence.
- [`paper_flow`](../flow/paper.md) uses the preserved source pages when assembling a canonical `Paper`.
- A future PageIndex adapter may consume the same ordered pages to propose and navigate a document tree.

Flattened Markdown remains a compatibility view produced from the preserved pages. It is not the primary parsing result.
