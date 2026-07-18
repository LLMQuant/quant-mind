"""Ingestion — QuantMind fetch+format + our own robust summarization.

Design note: QuantMind's *fetch + format* layer is solid (arxiv API + httpx
download + pymupdf PDF->markdown + trafilatura HTML->markdown), so we use it
directly. Its *structured Paper-tree extraction* (``paper_flow``) is brittle
under OpenAI structured output (it demands UUID node ids the LLM won't emit),
so we skip it and run our own summarizer instead. The corpus item therefore
carries: source metadata, the cleaned markdown, a structured summary, and an
embedding — everything the query layer needs, none of the fragile tree.

All entry points are async (the fetch/format layer is async); the MCP server
awaits them and the CLI wraps them in ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qm_mcp.config import (
    SYNTH_CONTEXT_CHAR_LIMIT,
    corpus_dir,
    load_secrets,
)
from qm_mcp.embed import embed_text, summarize_markdown
from qm_mcp.grpo_suitability import GrpoSuitabilityScorer
from qm_mcp.store import CorpusStore, make_id
from quantmind.preprocess.fetch import (
    Fetched,
    fetch_arxiv,
    fetch_url,
    read_local_file,
)
from quantmind.preprocess.format import html_to_markdown, pdf_to_markdown

_INGEST_LOG = "ingestion_log.jsonl"
_grpo_scorer = GrpoSuitabilityScorer()
# Cap the markdown we persist per item (text only, lives outside git).
_MARKDOWN_STORE_LIMIT = 400_000


# ── format helper ──────────────────────────────────────────────────────
async def _to_markdown(raw: Fetched) -> str:
    ct = (raw.content_type or "").lower()
    if ct.startswith("application/pdf"):
        return await pdf_to_markdown(raw.bytes)
    if ct.startswith("text/html"):
        return await html_to_markdown(
            raw.bytes.decode("utf-8", errors="replace")
        )
    return raw.bytes.decode("utf-8", errors="replace")


def _derive_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip() or fallback
        if s:
            return s[:160]
    return fallback


# ── ledger ─────────────────────────────────────────────────────────────
def _append_ingestion_log(record: dict[str, Any]) -> None:
    entry = {
        "id": record["id"],
        "title": record.get("title"),
        "source_type": record.get("source_type"),
        "source": record.get("source"),
        "ingested_at": record.get("ingested_at"),
        "grpo_suitability": record.get("grpo_suitability"),
        "event": "research.ingest",
    }
    with (corpus_dir() / _INGEST_LOG).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# ── core persist ───────────────────────────────────────────────────────
async def _persist(
    *,
    source_type: str,
    source: str,
    markdown: str,
    meta: dict[str, Any],
    store: CorpusStore,
    force: bool,
) -> dict[str, Any]:
    item_id = make_id(source_type, source)
    if store.exists(item_id) and not force:
        existing = store.get(item_id) or {}
        return {
            "id": item_id,
            "status": "exists",
            "title": existing.get("title"),
            "source_type": source_type,
            "source": source,
        }

    title_hint = meta.get("title") or _derive_title(markdown, fallback=source)
    structured = await asyncio.to_thread(
        summarize_markdown, title_hint, source, markdown
    )
    # arXiv metadata title is authoritative; otherwise prefer the LLM-recovered
    # title (it ignores library watermarks / download stamps that the
    # first-line heuristic would otherwise grab), then the heuristic hint.
    title = meta.get("title") or structured.get("title") or title_hint

    summary = structured["summary"]
    key_findings = structured["key_findings"]
    abstract = meta.get("abstract") or ""

    embed_blob = "\n".join(
        [title, abstract, summary, " ".join(key_findings)]
    ).strip()
    full_context = "\n\n".join(
        [
            f"# {title}",
            f"Abstract: {abstract}" if abstract else "",
            f"Summary: {summary}",
            "Key findings:\n- " + "\n- ".join(key_findings)
            if key_findings
            else "",
            "--- Source excerpt ---\n" + markdown,
        ]
    )[:SYNTH_CONTEXT_CHAR_LIMIT]

    as_of = meta.get("published_at") or datetime.now(timezone.utc).isoformat()
    record: dict[str, Any] = {
        "id": item_id,
        "source_type": source_type,
        "source": source,
        "item_type": "paper"
        if source_type in ("arxiv", "local")
        else "document",
        "title": title,
        "authors": list(meta.get("authors") or []),
        "arxiv_id": meta.get("arxiv_id"),
        "abstract": abstract,
        "summary": summary,
        "key_findings": key_findings,
        "tags": structured["tags"],
        "asset_classes": structured["asset_classes"],
        "categories": list(meta.get("categories") or []),
        "as_of": as_of,
        "embedding_text": embed_blob,
        "full_context": full_context,
        "markdown": markdown[:_MARKDOWN_STORE_LIMIT],
        "markdown_chars": len(markdown),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "grpo_suitability": _grpo_scorer.score_entry(
            {
                "source_type": source_type,
                "source": source,
                "markdown": markdown[:_MARKDOWN_STORE_LIMIT],
                "markdown_chars": len(markdown),
            }
        ),
    }

    vector = await asyncio.to_thread(embed_text, embed_blob)
    store.add(record, vector)
    _append_ingestion_log(record)
    return {
        "id": item_id,
        "status": "ingested",
        "title": title,
        "source_type": source_type,
        "source": source,
        "authors": record["authors"],
        "tags": record["tags"],
        "asset_classes": record["asset_classes"],
    }


# ── public entry points ────────────────────────────────────────────────
async def ingest_arxiv(
    arxiv_id: str, *, store: CorpusStore | None = None, force: bool = False
):
    load_secrets()
    store = store or CorpusStore()
    raw = await fetch_arxiv(arxiv_id)
    markdown = await pdf_to_markdown(raw.bytes)
    meta = {
        "title": raw.title,
        "authors": list(raw.authors),
        "arxiv_id": raw.arxiv_id,
        "abstract": raw.abstract,
        "published_at": raw.published_at.isoformat()
        if raw.published_at
        else None,
        "categories": list(raw.categories),
    }
    return await _persist(
        source_type="arxiv",
        source=raw.arxiv_id or arxiv_id,
        markdown=markdown,
        meta=meta,
        store=store,
        force=force,
    )


async def ingest_url(
    url: str, *, store: CorpusStore | None = None, force: bool = False
):
    load_secrets()
    store = store or CorpusStore()
    raw = await fetch_url(url)
    markdown = await _to_markdown(raw)
    return await _persist(
        source_type="url",
        source=url,
        markdown=markdown,
        meta={"content_type": raw.content_type},
        store=store,
        force=force,
    )


async def ingest_pdf(
    path: str, *, store: CorpusStore | None = None, force: bool = False
):
    load_secrets()
    store = store or CorpusStore()
    abspath = str(Path(path).expanduser().resolve())
    if not Path(abspath).is_file():
        return {
            "id": None,
            "status": "error",
            "error": f"file not found: {abspath}",
        }
    raw = await read_local_file(abspath)
    markdown = await _to_markdown(raw)
    return await _persist(
        source_type="local",
        source=abspath,
        markdown=markdown,
        meta={"content_type": raw.content_type},
        store=store,
        force=force,
    )


async def ingest_text(
    text: str,
    *,
    title_hint: str | None = None,
    store: CorpusStore | None = None,
    force: bool = False,
):
    load_secrets()
    store = store or CorpusStore()
    key = (
        (title_hint or "text")
        + ":"
        + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    )
    meta = {"title": title_hint} if title_hint else {}
    return await _persist(
        source_type="text",
        source=key,
        markdown=text,
        meta=meta,
        store=store,
        force=force,
    )
