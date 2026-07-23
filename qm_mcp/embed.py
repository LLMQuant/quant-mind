"""Embedding + answer-synthesis via OpenAI (the key QuantMind already needs).

Kept tiny and synchronous; callers on the async MCP path wrap these in
``asyncio.to_thread`` so the event loop is never blocked.
"""

from __future__ import annotations

import json

import numpy as np
from openai import OpenAI

from qm_mcp.config import (
    EMBED_CHAR_LIMIT,
    EMBED_MODEL,
    SYNTH_MODEL,
    require_openai_key,
)

# Markdown handed to the summarizer is truncated to keep the call cheap and
# inside context limits; the head of a paper carries abstract + intro +
# method, which is what the summary needs.
_SUMMARY_INPUT_CHAR_LIMIT = 30_000

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=require_openai_key())
    return _client


def embed_text(text: str) -> np.ndarray:
    """Embed one string -> float32 vector."""
    payload = (text or "").strip()[:EMBED_CHAR_LIMIT] or "(empty)"
    resp = _get_client().embeddings.create(model=EMBED_MODEL, input=payload)
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def synthesize_answer(question: str, contexts: list[dict]) -> str:
    """Answer ``question`` grounded in retrieved corpus contexts.

    ``contexts`` is a list of {title, source, text}. The model is told to
    answer ONLY from the provided material and to name the papers it used,
    so the corpus stays the source of truth (no free-floating hallucination).
    """
    if not contexts:
        return (
            "No matching items in the corpus yet — ingest some research first."
        )

    blocks = []
    for i, c in enumerate(contexts, 1):
        blocks.append(
            f"[{i}] {c.get('title', 'untitled')} "
            f"({c.get('source', '?')})\n{c.get('text', '')}"
        )
    corpus_block = "\n\n".join(blocks)

    system = (
        "You are a quantitative-finance research assistant answering from a "
        "private corpus of ingested papers. Answer ONLY from the provided "
        "sources. Cite the bracketed source numbers you use, e.g. [1]. If the "
        "corpus does not contain the answer, say so plainly rather than "
        "guessing."
    )
    user = f"Question: {question}\n\n--- Corpus sources ---\n{corpus_block}"

    resp = _get_client().chat.completions.create(
        model=SYNTH_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def summarize_markdown(title: str, source: str, markdown: str) -> dict:
    """Extract a structured research summary from markdown.

    Returns ``{summary, key_findings[], tags[], asset_classes[]}``. This is
    our own robust replacement for QuantMind's brittle ``Paper`` tree
    extraction (which demands UUID node ids the LLM won't reliably emit).
    Defensive: malformed JSON degrades to a plain-text summary.
    """
    body = (markdown or "").strip()[:_SUMMARY_INPUT_CHAR_LIMIT] or "(empty)"
    system = (
        "You extract a structured summary of a quantitative-finance research "
        "document for a searchable corpus. Respond with a JSON object with "
        "keys: title (the document's real title — recover it from the content; "
        "ignore library watermarks, download stamps, headers/footers), summary "
        "(a dense 150-250 word abstract covering the problem, method, and main "
        "result), key_findings (list of 3-7 concise bullet strings), tags (list "
        "of short topic tags), asset_classes (list, e.g. "
        "equities/futures/fx/crypto/rates, empty if unspecified). Be faithful "
        "to the source; do not invent results."
    )
    user = f"Title hint: {title}\nSource: {source}\n\n--- Document ---\n{body}"
    resp = _get_client().chat.completions.create(
        model=SYNTH_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "title": "",
            "summary": raw,
            "key_findings": [],
            "tags": [],
            "asset_classes": [],
        }
    return {
        "title": str(data.get("title", "")).strip(),
        "summary": str(data.get("summary", "")).strip(),
        "key_findings": [str(x) for x in (data.get("key_findings") or [])],
        "tags": [str(x) for x in (data.get("tags") or [])],
        "asset_classes": [str(x) for x in (data.get("asset_classes") or [])],
    }
