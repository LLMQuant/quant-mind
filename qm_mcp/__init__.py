"""qm_mcp — the research-corpus surface built on top of QuantMind ingestion.

QuantMind v0.2 ships ingestion + LLM extraction only (``paper_flow``); the
persistence, embedding, semantic-query, and MCP layers (its "Stage 2 /
Data MCP" vision) are not yet built upstream. This package supplies exactly
that missing layer so QuantMind becomes a usable, queryable corpus for
Thomas's trading + AVST research:

    ingest (paper_flow)  ->  CorpusStore (JSON + vectors)  ->  semantic query
                                      \\-> MCP server (Hermes / Dispatch / Conductor)

It is intentionally self-contained and dependency-light: it reuses
QuantMind's own venv (openai, numpy, pydantic, httpx, mcp) and stores the
corpus on the local filesystem under ``QM_CORPUS_DIR``.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
