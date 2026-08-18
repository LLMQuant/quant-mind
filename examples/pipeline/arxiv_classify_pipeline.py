"""End-to-end example: fetch arXiv papers, optional PDF parse, LLM classify, save JSON.

This example mirrors a script-like workflow while using QuantMind's modules:
- ArxivSource for fetching papers
- PDFParser for optional PDF -> text/markdown extraction
- LLMBlock (via LLMTaggerConfig/LLMConfig) for structured classification
- Result saver merges into a dated JSON file under data/
"""

import json
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from quantmind.config import ArxivSourceConfig, LLMConfig
from quantmind.parsers.pdf_parser import PDFParser
from quantmind.sources import ArxivSource
from quantmind.tagger.llm_tagger import LLMTagger
from quantmind.config.taggers import LLMTaggerConfig
from quantmind.models import Paper
from quantmind.utils.logger import get_logger


logger = get_logger(__name__)

ARXIV_ABS_URL = "https://arxiv.org/abs/"
PAPERSWITHCODE_BASE = "https://arxiv.paperswithcode.com/api/v0/papers/"


def _get_code_url(paper_id: str) -> Optional[str]:
    try:
        resp = requests.get(f"{PAPERSWITHCODE_BASE}{paper_id}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("official"):
            return data["official"].get("url")
    except Exception as e:
        logger.warning(f"Code URL fetch failed for {paper_id}: {e}")
    return None


def _classify_paper(
    llm_tagger: LLMTagger, paper: Paper, content_override: Optional[str] = None
) -> Dict[str, Any]:
    """Produce a structured classification matching the requested schema using LLMBlock."""
    # Build prompt
    content = content_override or paper.content or ""
    prompt = f"""Analyze this academic paper and return ONLY valid JSON with fields:
{{
  "trading_frequency": "string",
  "market_type": "string",
  "models_used": ["string"],
  "data_types": ["string"],
  "trading_strategies": ["string"]
}}

Title: {paper.title}
Abstract: {paper.abstract}
Content: {content[:6000]}

Constraints:
- Respond with JSON only, no prose.
- If uncertain, use "unknown" or empty lists.
"""

    # Use LLMBlock directly for structured output
    llm_block = llm_tagger.llm_block
    if llm_block is None:
        return {
            "trading_frequency": "unknown",
            "market_type": "unknown",
            "models_used": [],
            "data_types": [],
            "trading_strategies": [],
        }

    # Try strict response_format for OpenAI-compatible models
    response = llm_block.generate_structured_output(
        prompt,
        response_format={"type": "json_object"},
        max_tokens=800,
        temperature=0.0,
    )

    if isinstance(response, Dict):
        # Validate keys, fill defaults
        return {
            "trading_frequency": response.get("trading_frequency", "unknown"),
            "market_type": response.get("market_type", "unknown"),
            "models_used": response.get("models_used", []) or [],
            "data_types": response.get("data_types", []) or [],
            "trading_strategies": response.get("trading_strategies", []) or [],
        }

    # Fallback: try free-form then JSON-extract
    raw = llm_block.generate_text(prompt, max_tokens=800, temperature=0.0)
    if not raw:
        return {
            "trading_frequency": "unknown",
            "market_type": "unknown",
            "models_used": [],
            "data_types": [],
            "trading_strategies": [],
        }
    try:
        return json.loads(raw)
    except Exception:
        # Last resort: extract JSON from text
        parsed = llm_block._extract_json_from_text(raw)  # type: ignore[attr-defined]
        if isinstance(parsed, dict):
            return {
                "trading_frequency": parsed.get("trading_frequency", "unknown"),
                "market_type": parsed.get("market_type", "unknown"),
                "models_used": parsed.get("models_used", []) or [],
                "data_types": parsed.get("data_types", []) or [],
                "trading_strategies": parsed.get("trading_strategies", []) or [],
            }
        return {
            "trading_frequency": "unknown",
            "market_type": "unknown",
            "models_used": [],
            "data_types": [],
            "trading_strategies": [],
        }


def run_pipeline(
    topics_or_queries: Dict[str, Any],
    max_results: int = 10,
    download_and_parse_pdf: bool = False,
    output_dir: str = "data",
    llm_model: str = "gpt-4o",
) -> Path:
    """Run the pipeline and save merged results to data/arxiv_papers_YYYY-MM-DD.json.

    topics_or_queries accepts formats:
    - {"machine learning finance": {"filters": ["cat:q-fin.ST", "all:algorithmic trading"], "max_results": 5}}
    - {"algorithmic trading": {}}
    - {"quantitative finance": None}
    - {"cat:q-fin.ST": 8}  # value as custom max_results
    """

    # Initialize components
    arxiv_cfg = ArxivSourceConfig(max_results=max_results)
    arxiv_source = ArxivSource(config=arxiv_cfg)

    # pdf_parser = PDFParser({"method": "marker", "download_pdfs": True}) if download_and_parse_pdf else None

    # llm_cfg = LLMConfig(model=llm_model, temperature=0.0)
    # tagger_cfg = LLMTaggerConfig(llm_config=llm_cfg, max_tags=5)
    # llm_tagger = LLMTagger(config=tagger_cfg)

    all_results: Dict[str, Any] = {}

    # Iterate topics
    for topic, info in topics_or_queries.items():
        if isinstance(info, dict) and "filters" in info:
            query = " OR ".join(info.get("filters", [])) or topic
            topic_max = int(info.get("max_results", max_results))
        elif isinstance(info, int):
            query = topic
            topic_max = int(info)
        else:
            query = topic
            topic_max = max_results

        logger.info(f"Searching topic '{topic}' with query '{query}' (max={topic_max})")
        papers = arxiv_source.search(query=query, max_results=topic_max)

        for paper in papers:
            paper_id = paper.get_primary_id()

            # Enrich: code URL
            code_url = _get_code_url(paper_id)
            if code_url:
                paper.code_url = code_url

            # Optional: parse PDF to enrich content
            content_override = None
            # if pdf_parser:
            #     parsed = pdf_parser.parse_paper(paper)
            #     content_override = parsed.content if parsed and parsed.content else None

            # # Classify via LLM
            # classification = _classify_paper(llm_tagger, paper, content_override)

            # Build record similar to requested schema
            record = {
                "topic": topic,
                "title": paper.title,
                "authors": ", ".join(paper.authors),
                "first_author": ", ".join(paper.authors[:3]) if paper.authors else "",
                "abstract": (paper.abstract or "").replace("\n", " "),
                "url": f"{ARXIV_ABS_URL}{paper_id}",
                "code_url": paper.code_url,
                "category": (paper.meta_info or {}).get("primary_category"),
                "publish_time": paper.published_date.date().isoformat() if paper.published_date else None,
                "update_time": None,
                "comments": (paper.meta_info or {}).get("comment", "") or "",
                # "classification": classification,
            }

            all_results[paper_id] = record

    # Save merged results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    out_path = out_dir / f"arxiv_papers_{today}.json"

    existing: Dict[str, Any] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    existing.update(all_results)
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(all_results)} papers to {out_path}")
    return out_path


if __name__ == "__main__":
    # Minimal demo
    topics = {
        "cat:q-fin.ST": 5,
        "machine learning finance": {"filters": ["cat:q-fin.ST", 'all:"algorithmic trading"'], "max_results": 3},
    }
    run_pipeline(topics_or_queries=topics, max_results=5, download_and_parse_pdf=False)

