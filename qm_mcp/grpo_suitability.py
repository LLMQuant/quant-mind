"""GRPO-suitability scoring for QuantMind corpus entries.

Implements the weak-vs-strong discrimination-gap framework from Autodata
(Kulikov et al., arXiv:2606.25996, FAIR at Meta, 2026). In v1 the scorer
is a pure heuristic — no live model calls — with documented hooks for the
v2 solver-gap measurement once QuantMind has the LLM substrate for it.

Scoring bands
─────────────
    high   — long document + authoritative source + technical content
              (strong proxy for "learnable zone" per Autodata §2b)
    medium  — everything else
    low    — short document + low-authority source + no code
              (strong proxy for "too easy / too sparse to discriminate")
"""

from __future__ import annotations

import re
from typing import Any

# Thresholds for the length proxy.
_SHORT_CHARS = 5_000
_LONG_CHARS = 20_000

# Source-type → domain-band mapping.
_DOMAIN_BAND_BY_SOURCE_TYPE: dict[str, str] = {
    "arxiv": "arxiv",
    "local": "arxiv",  # local PDFs are almost always papers
}

# URL substring → domain band (checked in priority order).
_URL_DOMAIN_PATTERNS: list[tuple[str, str]] = [
    ("arxiv.org", "arxiv"),
    ("ssrn.com", "ssrn"),
    ("papers.ssrn.com", "ssrn"),
    ("substack.com", "substack"),
]

# Regex for fenced code blocks and inline code of ≥4 chars.
_CODE_PATTERN = re.compile(r"```|\$\$|`[^`]{4,}`")


class GrpoSuitabilityScorer:
    """Deterministic v1 GRPO-suitability scorer.

    Operates entirely on fields already present in a corpus entry dict.
    No network calls, no LLM inference — pure heuristic.

    V2 plan (when QuantMind has the LLM substrate):
    ────────────────────────────────────────────────
    # TODO(v2): replace heuristic with Autodata-style solver gap.
    # Steps:
    #   1. _weak_query(entry)  → float  (surface-recall query via cheap model)
    #   2. _strong_query(entry) → float  (application query via frontier model)
    #   3. gap = _strong_query - _weak_query
    #   4. if gap >= 0.20 and _strong_query >= 0.65: return "high"
    #      elif gap >= 0.10:                          return "medium"
    #      else:                                       return "low"
    # Acceptance criterion per Autodata Table 1:
    #   strong avg ≥ 0.65, weak avg < 0.50, gap ≥ 20pp → "high"
    """

    # ── public API ────────────────────────────────────────────────────────

    def score_entry(self, entry: dict[str, Any]) -> str:
        """Return 'high', 'medium', or 'low' for *entry*.

        The entry dict must at minimum contain 'source_type'. All other
        fields (markdown, markdown_chars, source) are read with .get() so
        the scorer is safe on partial records and legacy entries.
        """
        lb = self._length_band(entry)
        db = self._domain_band(entry)
        cp = self._code_present(entry)

        # V1 deterministic rule — see module docstring.
        if lb == "long" and db == "arxiv" and cp:
            return "high"
        if lb == "short" and db == "news" and not cp:
            return "low"
        return "medium"

    # ── private helpers ───────────────────────────────────────────────────

    def _length_band(self, entry: dict[str, Any]) -> str:
        """Classify entry length as 'short', 'medium', or 'long'."""
        chars = entry.get("markdown_chars")
        if chars is None:
            markdown = entry.get("markdown") or entry.get("full_context") or ""
            chars = len(markdown)
        if chars < _SHORT_CHARS:
            return "short"
        if chars >= _LONG_CHARS:
            return "long"
        return "medium"

    def _domain_band(self, entry: dict[str, Any]) -> str:
        """Classify source authority as 'arxiv', 'ssrn', 'substack', or 'news'."""
        source_type = entry.get("source_type") or ""
        if source_type in _DOMAIN_BAND_BY_SOURCE_TYPE:
            return _DOMAIN_BAND_BY_SOURCE_TYPE[source_type]

        source = (entry.get("source") or "").lower()
        for substring, band in _URL_DOMAIN_PATTERNS:
            if substring in source:
                return band

        # text ingests and unknown URL sources default to "news" (lowest authority).
        return "news"

    def _code_present(self, entry: dict[str, Any]) -> bool:
        """Return True if the entry's markdown contains code or math blocks."""
        markdown = entry.get("markdown") or entry.get("full_context") or ""
        return bool(_CODE_PATTERN.search(markdown))
