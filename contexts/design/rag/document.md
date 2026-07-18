# Retrieve page-aware document evidence with LlamaIndex

## Quick Summary

- **Purpose**: Define how an ordered `ParsedDocument` becomes deterministic page-aware chunks and ranked document-local evidence.
- **Read when**: Changing document chunking, source spans, document-local retrieval, Paper Flow chunk construction, or a future PageIndex adapter.
- **Status**: Implemented by `quantmind.rag.document`.
- **Core rule**: LlamaIndex owns splitting and ranking mechanics; QuantMind owns stable identity and source/page provenance.

## Contents

- [Package Boundary](#package-boundary)
- [Chunk Contract](#chunk-contract)
- [Document-Local Retrieval](#document-local-retrieval)
- [Paper Flow Boundary](#paper-flow-boundary)
- [Collection Search and PageIndex](#collection-search-and-pageindex)
- [What This Package Does Not Abstract](#what-this-package-does-not-abstract)

## Package Boundary

`quantmind.preprocess` owns deterministic source parsing and returns a page-aware [`ParsedDocument`](../preprocess/pdf.md). `quantmind.rag` may import that value and apply LlamaIndex transformations and retrieval. Preprocessing never imports RAG, so parsing remains usable without a query or index.

LlamaIndex is a required dependency and owns `SentenceSplitter`, BM25, nodes, indexes, retrievers, ranking mechanics, and upstream parameters. QuantMind adds stable source hashes, page ownership, page-local character spans, block coordinates, screenshot/image references, and conversion back to typed evidence.

## Chunk Contract

`chunk_parsed_document()` applies LlamaIndex `SentenceSplitter` independently to each non-empty physical page. Splitting by page prevents a chunk from erasing page ownership or spanning an implicit page boundary.

`SentenceSplitterConfig` exposes chunk size, overlap, separator, paragraph separator, and tokenizer behavior by their upstream meanings. It does not reimplement the splitter.

Each `ParsedChunk` records:

- deterministic SHA-256-derived chunk ID;
- exact document source hash;
- 1-based physical page number;
- page-local `start_char` and `end_char` offsets;
- exact chunk text;
- overlapping parser block bounding boxes;
- page screenshot and extracted-image paths.

Repeating chunking for the same parsed document and configuration produces the same ordered texts, spans, and IDs. Empty pages remain in `ParsedDocument` but produce no chunks.

## Document-Local Retrieval

`retrieve_parsed_document()` chunks one document, uses LlamaIndex BM25, and returns ranked `ParsedDocumentHit` values. Each hit wraps one `ParsedChunk` and a score. The operation is transient and requires no canonical library write.

LlamaIndex `Document`, node, retriever, index, and score-wrapper types remain private. Public callers receive frozen QuantMind dataclasses with enough evidence to trace a hit back to physical pages and parser artifacts.

## Paper Flow Boundary

[`paper_flow`](../flow/paper.md) uses `chunk_parsed_document()` as the deterministic split stage. It converts `ParsedChunk` values into canonical `PaperChunk` members only after an exact `PaperSourceRevision` exists.

The conversion replaces parser paths with canonical source asset IDs and validates character spans against page evidence. The resulting `PaperChunkSet` is a durable, independently versioned artifact. `quantmind.rag` itself does not import or construct canonical paper models.

## Collection Search and PageIndex

Document-local RAG and collection search have different responsibilities. [`LocalKnowledgeLibrary`](../library/local.md) stores canonical sources and artifacts in SQLite and privately uses LlamaIndex for collection-wide embedding ranking. It does not persist transient `ParsedDocumentHit` values.

Paper Flow V1 defines no paper tree. Under `quantmind.rag`, a future PageIndex helper is limited to a stateless document-local operation: it may consume `ParsedDocument` and return a bounded draft outline or navigation evidence, because `rag` may import only `preprocess`. Building a persisted canonical tree and running library-backed agentic navigation sit above `library` and are owned by `quantmind.flows` and `quantmind.mind`. See [Build and navigate page-preserving knowledge trees](../mind/navigation.md). Canonical IDs, links, citations, and source-backed text remain code-owned throughout.

## What This Package Does Not Abstract

The package does not define a public `Retriever`, `VectorStore`, backend registry, provider protocol, generic query engine, answer-synthesis framework, or canonical paper tree. Add another direct opinionated operation only when a real pipeline needs it; do not add a wrapper solely to hide an upstream LlamaIndex call.
