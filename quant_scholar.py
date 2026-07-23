#!/usr/bin/env python3
"""quant_scholar.py — Daily arxiv paper fetcher for quantitative finance.

Fetches papers published in the last 7 days from:
  - q-fin.*       (all Quantitative Finance sub-categories, primary)
  - cs.LG / cs.AI (filtered to papers with quant-finance keywords)
  - stat.ML       (filtered to papers with quant-finance keywords)

Ranks by keyword-match count + primary-category bonus, takes top 50,
groups by topic, and writes:
  docs/papers.md          — markdown table grouped by topic
  docs/quant-scholar.json — JSON array of structured paper records
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────

N_MAX = 50
WINDOW_DAYS = 7

# Keywords for (a) filtering cs/stat papers and (b) scoring all papers.
QUANT_KEYWORDS: list[str] = [
    "trading",
    "portfolio",
    "market microstructure",
    "financial market",
    "alpha",
    "factor model",
    "momentum",
    "volatility",
    "order flow",
    "execution",
    "risk management",
    "hedge",
    "arbitrage",
    "option pricing",
    "asset allocation",
    "equity",
    "stock",
    "futures",
    "high frequency",
    "quantitative finance",
    "investment strategy",
    "backtesting",
]

# (query, needs_keyword_filter)  — primary q-fin.* fetched without filter
SEARCHES: list[tuple[str, bool]] = [
    ("cat:q-fin.*", False),
    ("cat:cs.LG", True),
    ("cat:cs.AI", True),
    ("cat:stat.ML", True),
]

# Topic classification: first match wins; applied to title + abstract (lowered)
TOPICS: list[tuple[str, list[str]]] = [
    (
        "Reinforcement Learning in Finance",
        [
            "reinforcement learning",
            "q-learning",
            "policy gradient",
            "markov decision process",
            "actor-critic",
            "deep q-network",
        ],
    ),
    (
        "Deep Learning in Finance",
        [
            "deep learning",
            "transformer",
            "attention mechanism",
            "lstm",
            "bert",
            "gpt",
            "autoencoder",
            "graph neural network",
            "convolutional neural network",
            "neural network",
        ],
    ),
    (
        "Time Series Forecasting",
        [
            "time series",
            "forecasting",
            "arima",
            "temporal convolution",
            "state space model",
            "sequence prediction",
        ],
    ),
    (
        "Machine Learning in Finance",
        [
            "machine learning",
            "gradient boost",
            "random forest",
            "xgboost",
            "lightgbm",
            "support vector",
            "classification",
            "ensemble",
        ],
    ),
]
DEFAULT_TOPIC = "Quantitative Finance"

# ── helpers ───────────────────────────────────────────────────────────────────


def _arxiv_id(r: arxiv.Result) -> str:
    """Return version-stripped arxiv ID, e.g. '2506.12345'."""
    raw = r.entry_id.split("/abs/")[-1]
    return re.sub(r"v\d+$", "", raw)


def _short_authors(r: arxiv.Result) -> str:
    names = [a.name for a in r.authors]
    if len(names) <= 2:
        return ", ".join(names)
    return f"{names[0]}, {names[1]} et.al."


def _has_quant_keyword(r: arxiv.Result) -> bool:
    text = (r.title + " " + r.summary).lower()
    return any(kw in text for kw in QUANT_KEYWORDS)


def _score(r: arxiv.Result, is_primary: bool) -> float:
    text = (r.title + " " + r.summary).lower()
    match_count = sum(1 for kw in QUANT_KEYWORDS if kw in text)
    return float(match_count) + (5.0 if is_primary else 0.0)


def _categorize(r: arxiv.Result) -> str:
    text = (r.title + " " + r.summary).lower()
    for topic, keywords in TOPICS:
        if any(kw in text for kw in keywords):
            return topic
    return DEFAULT_TOPIC


# ── fetching ──────────────────────────────────────────────────────────────────


def _fetch(
    query: str, cutoff: datetime, max_results: int = 300
) -> list[arxiv.Result]:
    client = arxiv.Client(page_size=50, delay_seconds=3.0)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results: list[arxiv.Result] = []
    for r in client.results(search):
        pub = r.published
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        if pub < cutoff:
            break
        results.append(r)
    return results


def collect_papers() -> list[tuple[arxiv.Result, float, str]]:
    """Return list of (result, score, topic) for the top N papers."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    seen: set[str] = set()
    pool: list[tuple[arxiv.Result, float, str]] = []

    for query, needs_filter in SEARCHES:
        log.info("Fetching %s ...", query)
        is_primary = not needs_filter
        batch = _fetch(query, cutoff)
        log.info("  -> %d papers within window", len(batch))
        for r in batch:
            aid = _arxiv_id(r)
            if aid in seen:
                continue
            if needs_filter and not _has_quant_keyword(r):
                continue
            seen.add(aid)
            pool.append((r, _score(r, is_primary), _categorize(r)))

    pool.sort(key=lambda x: x[1], reverse=True)
    return pool[:N_MAX]


# ── rendering ─────────────────────────────────────────────────────────────────

_TABLE_HEADER = (
    "| 📅 Publish Date | 📖 Title | 👨‍💻 Authors | 🔗 PDF | 💻 Code | 💬 Comment | 📜 Abstract |\n"
    "|:--------------:|:----------------------------|:------------------|:------:|:------:|:-------:|:--------|"
)


def render_markdown(papers: list[tuple[arxiv.Result, float, str]]) -> str:
    """Render scored papers to a GitHub-flavoured markdown table grouped by topic."""
    today = datetime.now(timezone.utc).strftime("%Y.%m.%d")

    groups: dict[str, list[tuple[arxiv.Result, float, str]]] = {}
    for item in papers:
        groups.setdefault(item[2], []).append(item)

    topic_order = [t for t, _ in TOPICS if t in groups]
    for t in groups:
        if t not in topic_order:
            topic_order.append(t)

    def anchor(topic: str) -> str:
        return "#-" + topic.lower().replace(" ", "-")

    toc_items = "\n".join(
        f"    <li><a href={anchor(t)}>📌 {t}</a></li>" for t in topic_order
    )

    lines: list[str] = [
        '<p align="center"><h1 align="center">🌟 QUANT-SCHOLAR 🌟</h1>'
        '<h2 align="center">Automatically Quantitative Finance Papers List</h2></p>',
        '<p align="center"><img src="https://raw.githubusercontent.com/LLMQuant/quant-scholar/main/asset/icon.png" width="180"></p>',
        "",
        f"## 🚩 Updated on {today}",
        "<details>",
        "  <summary><strong>📜 Contents</strong></summary>",
        "  <ol>",
        toc_items,
        "  </ol>",
        "</details>",
        "",
    ]

    for topic in topic_order:
        lines.append(f"## 📌 {topic}")
        lines.append("")
        lines.append(_TABLE_HEADER)
        for r, _score_val, _topic in groups[topic]:
            aid = _arxiv_id(r)
            date_str = r.published.strftime("%Y-%m-%d")
            title = r.title.replace("|", "\\|").replace("\n", " ").strip()
            authors = _short_authors(r).replace("|", "\\|")
            pdf_link = f"[{aid}](http://arxiv.org/abs/{aid})"
            abstract = r.summary.replace("\n", " ").replace("|", "\\|").strip()
            abstract_cell = (
                f"<details><summary>Abstract (click to expand)</summary>"
                f"{abstract}</details>"
            )
            lines.append(
                f"| {date_str} | {title} | {authors} | {pdf_link} |  |  | {abstract_cell} |"
            )
        lines.append("")

    return "\n".join(lines)


def render_json(papers: list[tuple[arxiv.Result, float, str]]) -> str:
    """Render scored papers to a JSON array of structured records."""
    records = []
    for r, score, topic in papers:
        aid = _arxiv_id(r)
        records.append(
            {
                "title": r.title.replace("\n", " ").strip(),
                "authors": [a.name for a in r.authors],
                "arxiv_id": aid,
                "date": r.published.strftime("%Y-%m-%d"),
                "categories": r.categories,
                "abstract": r.summary.replace("\n", " ").strip(),
                "pdf_url": f"http://arxiv.org/pdf/{aid}",
                "score": score,
                "topic": topic,
            }
        )
    return json.dumps(records, ensure_ascii=False, indent=2)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Fetch, rank, and write recent quant-finance papers to docs/."""
    papers = collect_papers()
    log.info("Selected %d papers total", len(papers))

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    md_path = docs_dir / "papers.md"
    json_path = docs_dir / "quant-scholar.json"

    md_path.write_text(render_markdown(papers), encoding="utf-8")
    log.info("Wrote %s", md_path)

    json_path.write_text(render_json(papers), encoding="utf-8")
    log.info("Wrote %s", json_path)

    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    log.info("JSON validated: %d records", len(data))

    log.info("Done.")


if __name__ == "__main__":
    main()
