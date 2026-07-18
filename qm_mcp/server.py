"""QuantMind research-corpus MCP server (stdio).

Exposes the corpus to any MCP client — Personal Hermes (the #research
channel), Dispatch sessions, the Conductor, future Akazi AVST. Tools are the
contract the brief asked for:

    qm_ingest_url(url)          ingest a web page / hosted PDF
    qm_ingest_pdf(path)         ingest a local PDF / HTML / markdown file
    qm_ingest_arxiv(arxiv_id)   ingest an arXiv paper by id or URL
    qm_ingest_text(text,title)  ingest pasted text
    qm_query(question, k, distance_threshold)  natural-language query (grounded answer + sources)
    qm_list_corpus()            list everything ingested
    qm_delete_item(item_id)     remove one item

Run (under QuantMind's venv)::

    /Users/thomasadair/projects/quant-mind/.venv/bin/python -m qm_mcp.server
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from qm_mcp import ingest as _ingest
from qm_mcp.query import query as _query
from qm_mcp.store import CorpusStore

mcp = FastMCP("quantmind-research")


@mcp.tool()
async def qm_ingest_arxiv(arxiv_id: str) -> dict[str, Any]:
    """Ingest an arXiv paper into the research corpus.

    Args:
        arxiv_id: arXiv id (e.g. "1105.3115") or a full arxiv.org URL.
    """
    return await _ingest.ingest_arxiv(arxiv_id)


@mcp.tool()
async def qm_ingest_url(url: str) -> dict[str, Any]:
    """Ingest a web page or hosted PDF (news, blog, report) by URL."""
    return await _ingest.ingest_url(url)


@mcp.tool()
async def qm_ingest_pdf(path: str) -> dict[str, Any]:
    """Ingest a local PDF / HTML / Markdown file by filesystem path."""
    return await _ingest.ingest_pdf(path)


@mcp.tool()
async def qm_ingest_text(text: str, title: str | None = None) -> dict[str, Any]:
    """Ingest pasted raw text as a corpus item."""
    return await _ingest.ingest_text(text, title_hint=title)


@mcp.tool()
async def qm_query(
    question: str,
    k: int = 6,
    distance_threshold: float = 0.7,
) -> dict[str, Any]:
    """Ask the corpus a natural-language question.

    Returns a grounded answer (cited to ingested sources) plus the top-k
    matching items. Chunks with cosine distance > ``distance_threshold`` are
    filtered; returns empty sources when no candidate clears the threshold.
    """
    return await _query(question, k=k, distance_threshold=distance_threshold)


@mcp.tool()
def qm_list_corpus() -> dict[str, Any]:
    """List every item in the research corpus (metadata only)."""
    store = CorpusStore()
    items = store.list_records(light=True)
    return {"count": len(items), "items": items}


@mcp.tool()
def qm_delete_item(item_id: str) -> dict[str, Any]:
    """Delete one corpus item by its id."""
    removed = CorpusStore().delete(item_id)
    return {"id": item_id, "deleted": removed}


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
