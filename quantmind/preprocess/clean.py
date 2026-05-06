"""Pure text-cleaning helpers.

These run in-process (sync) — they're cheap byte-pushing operations that
do not warrant the ``asyncio.to_thread`` ceremony of the format layer.
Each function takes ``str`` and returns ``str``; callers can compose them
in whatever order suits their use case.
"""

import re
import unicodedata

# Map of common ligatures + smart quotes -> ASCII equivalents. Conservative
# on purpose: only replacements that make text searchable / model-friendly
# without destroying meaning (e.g. we don't strip diacritics).
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",  # non-breaking space
}

_LIGATURE_RE = re.compile("|".join(re.escape(k) for k in _LIGATURE_MAP))
_HORIZONTAL_WS_RE = re.compile(r"[ \t\f\v]+")
_TRIPLE_NEWLINE_RE = re.compile(r"\n{3,}")
# Control characters except for newline + tab.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_unicode(text: str) -> str:
    """Apply NFKC + ligature/smart-quote normalisation + control-char drop.

    Order matters: NFKC first (so e.g. fullwidth digits collapse to ASCII),
    then targeted replacements for characters NFKC leaves alone, then drop
    control characters that PDF extraction commonly leaks.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _LIGATURE_RE.sub(
        lambda m: _LIGATURE_MAP[m.group(0)], normalized
    )
    normalized = _CONTROL_RE.sub("", normalized)
    return normalized


def collapse_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace; keep paragraph breaks.

    Preserves single and double newlines (the markdown-paragraph signal)
    but collapses 3+ consecutive newlines to 2 so excessive blank gaps from
    PDF extraction disappear.
    """
    if not text:
        return ""
    collapsed = _HORIZONTAL_WS_RE.sub(" ", text)
    collapsed = _TRIPLE_NEWLINE_RE.sub("\n\n", collapsed)
    # Strip trailing whitespace on each line without touching empties.
    lines = [line.rstrip() for line in collapsed.split("\n")]
    return "\n".join(lines).strip()


def dedupe_lines(text: str) -> str:
    """Drop consecutive duplicate lines.

    Useful for killing repeated PDF page headers/footers ("Page 3 of 12",
    journal banners) without touching legitimate prose. Comparison is
    whitespace-insensitive — leading/trailing space on either copy still
    counts as a duplicate.
    """
    if not text:
        return ""
    output: list[str] = []
    last: str | None = None
    for line in text.split("\n"):
        key = line.strip()
        if key and key == last:
            continue
        output.append(line)
        last = key
    return "\n".join(output)
