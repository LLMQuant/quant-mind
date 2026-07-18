# Retrieve page-aware document evidence with LlamaIndex

## Quick Summary

- **Purpose**: Define how an ordered `ParsedDocument` becomes chunks and ranked evidence without leaking LlamaIndex types.
- **Read when**: Changing document chunking, document-local retrieval, RAG evidence, or a future PageIndex adapter.
- **Status**: Implemented by `quantmind.rag.document`.
- **Core rule**: `quantmind.rag` is an opinionated LlamaIndex data-plane package, not a generic retriever or backend framework.

## Contents

- [Package boundary](#package-boundary)
- [Chunk and retrieval contract](#chunk-and-retrieval-contract)
- [What this package does not abstract](#what-this-package-does-not-abstract)
- [Collection search and PageIndex](#collection-search-and-pageindex)

## Package boundary

`quantmind.preprocess` owns deterministic source parsing and returns a page-aware [`ParsedDocument`](../preprocess/pdf.md). `quantmind.rag` may import that value and apply LlamaIndex transformations and retrieval. Preprocessing never imports RAG, so parsing remains usable without a query or index.

LlamaIndex is a required dependency and owns the chunker, nodes, indexes, retrievers, ranking mechanics, and their supported parameters. QuantMind adds only the work that LlamaIndex cannot own: stable source hashes, page ownership, block coordinates, screenshot/image references, and conversion back to typed QuantMind evidence.

## Chunk and retrieval contract

`chunk_parsed_document()` applies LlamaIndex `SentenceSplitter` to each non-empty page without erasing physical page boundaries. `SentenceSplitterConfig` exposes the selected upstream parameters by their upstream meaning rather than reimplementing the algorithm.

Each `ParsedChunk` retains the exact source hash, 1-based page number, available block bounding boxes, and screenshot/image references. `retrieve_parsed_document()` uses LlamaIndex BM25 and returns ranked `ParsedDocumentHit` values. LlamaIndex `Document`, node, retriever, index, and score wrapper types remain private implementation details.

## What this package does not abstract

The package does not define a public `Retriever`, `VectorStore`, backend registry, provider protocol, query engine, or answer-synthesis framework. Add another direct, opinionated operation only when a real pipeline needs it. Do not build an abstraction solely to hide an upstream LlamaIndex call.

## Collection search and PageIndex

Document-local RAG and collection search have different responsibilities. [`LocalKnowledgeLibrary`](../library/local.md) stores canonical knowledge in SQLite and privately uses LlamaIndex for collection-wide semantic ranking. It does not own transient PDF parsing or document-local BM25 chunks.

A future PageIndex implementation may live under `quantmind.rag` as another opinionated document operation. It can consume `ParsedDocument` and return a limited tree draft or navigation evidence, while canonical IDs, links, citations, and source-backed text remain owned by QuantMind code. PageIndex does not have to run through `LocalKnowledgeLibrary.search()` or LlamaIndex vector ranking.
