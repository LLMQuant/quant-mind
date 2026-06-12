"""Configuration + secret loading for the QuantMind corpus surface.

Secrets are NOT hard-coded here. The OpenAI key (used by QuantMind's
``paper_flow`` extraction and by our embedding/synthesis calls) is loaded
from the canonical Hermes gateway env file ``~/.hermes/.env`` if present,
then from the process environment. This mirrors the Phase 3 Doppler/`.env`
pattern: the running gateway already owns these secrets.
"""

from __future__ import annotations

import os
from pathlib import Path

# Canonical secret source: the always-on Hermes gateway env file.
_HERMES_ENV = Path.home() / ".hermes" / ".env"

# Embedding + synthesis models. text-embedding-3-small is 1536-dim, cheap,
# and good enough for a coarse semantic pre-filter over a research corpus.
EMBED_MODEL = os.environ.get("QM_EMBED_MODEL", "text-embedding-3-small")
SYNTH_MODEL = os.environ.get("QM_SYNTH_MODEL", "gpt-4o-mini")
# Extraction model for paper_flow. gpt-4o-mini keeps per-paper cost to cents.
EXTRACT_MODEL = os.environ.get("QM_EXTRACT_MODEL", "gpt-4o-mini")

# Embedding input ceiling (chars). text-embedding-3-small caps at ~8191
# tokens; ~24k chars (~6k tokens) leaves comfortable headroom.
EMBED_CHAR_LIMIT = 24_000
# Synthesis context ceiling (chars) across all retrieved sources.
SYNTH_CONTEXT_CHAR_LIMIT = 14_000


def corpus_dir() -> Path:
    """Root directory for the persisted corpus (items + vectors)."""
    raw = os.environ.get("QM_CORPUS_DIR")
    base = (
        Path(raw).expanduser()
        if raw
        else (Path.home() / ".quantmind" / "corpus")
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_secrets() -> None:
    """Load OPENAI_API_KEY (and friends) from ~/.hermes/.env into os.environ.

    Existing process-env values win — we only fill gaps. This never prints
    or returns the secret value.
    """
    if not _HERMES_ENV.is_file():
        return
    try:
        for line in _HERMES_ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        # Secret file unreadable — fall through to whatever is already in
        # the environment. The OpenAI client will raise a clear error if the
        # key is genuinely absent.
        pass

    # CRITICAL: Hermes' OPENAI_API_KEY is an OpenRouter key (sk-or-...). That
    # 401s against api.openai.com and OpenRouter exposes no embeddings
    # endpoint. The real platform.openai.com key is stored separately as
    # VOICE_TOOLS_OPENAI_KEY (used for Whisper). Force it as the OpenAI key
    # for THIS process only so both QuantMind's openai-agents extraction and
    # our embeddings/synthesis hit real OpenAI. We also clear any OpenAI base
    # URL so the client cannot be redirected to OpenRouter.
    real = os.environ.get("VOICE_TOOLS_OPENAI_KEY", "").strip()
    if real:
        os.environ["OPENAI_API_KEY"] = real
        os.environ.pop("OPENAI_BASE_URL", None)


def require_openai_key() -> str:
    """Return the real OpenAI key or raise a clear, actionable error."""
    load_secrets()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "No OpenAI key available. QuantMind ingestion + corpus embedding "
            "need a real platform.openai.com key. Set VOICE_TOOLS_OPENAI_KEY "
            "(preferred) or OPENAI_API_KEY in ~/.hermes/.env."
        )
    if key.startswith("sk-or-"):
        raise RuntimeError(
            "The active OpenAI key is an OpenRouter key (sk-or-...), which "
            "cannot do embeddings or reach api.openai.com. Set "
            "VOICE_TOOLS_OPENAI_KEY to a real platform.openai.com key."
        )
    return key
