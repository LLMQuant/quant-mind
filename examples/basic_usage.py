#!/usr/bin/env python3
"""Basic usage example for QuantMind Stage 1 architecture.

This example demonstrates how to use the new QuantMind architecture to:
1. Set up sources, parsers, taggers, and storage
2. Create and execute a knowledge extraction pipeline
3. Process financial research papers from arXiv
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import quantmind
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantmind.workflow.agent import WorkflowAgent
from quantmind.sources.arxiv_source import ArxivSource
from quantmind.parsers.pdf_parser import PDFParser
from quantmind.tagger.rule_tagger import RuleTagger
from quantmind.tagger.llm_tagger import LLMTagger
from quantmind.storage.json_storage import JSONStorage
from quantmind.config.settings import create_default_config
from quantmind.utils.logger import setup_logger, get_logger

# Set up logging
setup_logger(level=20)  # INFO level
logger = get_logger(__name__)


def main():
    """Run the basic QuantMind usage example."""
    logger.info("Starting QuantMind basic usage example")

    # 1. Create workflow agent
    agent = WorkflowAgent(
        config={
            "max_workers": 2,
            "retry_attempts": 2,
            "timeout": 180,
            "enable_deduplication": True,
        }
    )

    # 2. Register components
    logger.info("Registering components...")

    # Register ArXiv source
    arxiv_source = ArxivSource(
        config={"max_results": 50, "sort_by": "SubmittedDate"}
    )
    agent.register_source("arxiv", arxiv_source)

    # Register PDF parser (optional, for full text extraction)
    pdf_parser = PDFParser(
        config={
            "method": "pymupdf",  # Use PyMuPDF for simplicity
            "download_pdfs": False,  # Skip PDF download for this example
            "max_file_size": 10,  # MB
        }
    )
    agent.register_parser("pdf", pdf_parser)

    # Register rule-based tagger
    rule_tagger = RuleTagger(config={"case_sensitive": False})
    agent.register_tagger("rule", rule_tagger)

    # Register LLM tagger (optional, requires OpenAI API key)
    if os.getenv("OPENAI_API_KEY"):
        llm_tagger = LLMTagger(
            config={
                "model_type": "openai",
                "model_name": "gpt-4",
                "temperature": 0.0,
            }
        )
        agent.register_tagger("llm", llm_tagger)
        logger.info("LLM tagger registered (OpenAI API key found)")
    else:
        logger.info("LLM tagger skipped (no OpenAI API key)")

    # Register JSON storage
    json_storage = JSONStorage(
        config={
            "storage_dir": "./data/quantmind_example",
            "auto_backup": True,
            "max_backup_count": 3,
        }
    )
    agent.register_storage("json", json_storage)

    # 3. Run quick extraction example
    logger.info(
        "Running quick extraction for machine learning in finance papers..."
    )

    try:
        papers = agent.run_quick_extraction(
            source_name="arxiv",
            query="cat:q-fin.ST OR cat:q-fin.TR OR (machine learning AND finance)",
            max_papers=10,
            tagger_name="rule",
        )

        logger.info(f"Successfully extracted {len(papers)} papers")

        # 4. Display results
        print("\n" + "=" * 80)
        print("EXTRACTION RESULTS")
        print("=" * 80)

        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper.title}")
            print(
                f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}"
            )
            print(f"   Categories: {', '.join(paper.categories)}")
            print(
                f"   Tags: {', '.join(paper.tags[:5])}{'...' if len(paper.tags) > 5 else ''}"
            )
            print(f"   ArXiv ID: {paper.arxiv_id or 'N/A'}")
            print(
                f"   Published: {paper.published_date.strftime('%Y-%m-%d') if paper.published_date else 'N/A'}"
            )
            print(f"   Abstract: {paper.abstract[:200]}...")

    except Exception as e:
        logger.error(f"Quick extraction failed: {e}")
        return

    # 5. Create and execute a full pipeline
    logger.info("\nCreating full extraction pipeline...")

    try:
        pipeline = agent.create_extraction_pipeline(
            name="finance_ml_pipeline",
            source_name="arxiv",
            query="cat:q-fin.ST AND machine learning",
            max_papers=5,
            tagger_name="rule",
            storage_name="json",
        )

        logger.info("Executing pipeline...")
        results = agent.execute_pipeline("finance_ml_pipeline")

        print("\n" + "=" * 80)
        print("PIPELINE RESULTS")
        print("=" * 80)

        for task_id, result in results.items():
            print(f"\nTask {task_id}: {type(result).__name__}")
            if hasattr(result, "__len__"):
                print(f"  Results count: {len(result)}")

        # Get pipeline statistics
        stats = agent.get_pipeline_status("finance_ml_pipeline")
        if stats:
            print(f"\nPipeline Statistics:")
            print(f"  Status: {stats['status']}")
            print(f"  Total tasks: {stats['total_tasks']}")
            print(f"  Duration: {stats.get('duration', 'N/A')} seconds")
            print(f"  Task counts: {stats['task_counts']}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return

    # 6. Storage examples
    logger.info("\nTesting storage operations...")

    try:
        # Get storage statistics
        storage_info = json_storage.get_storage_info()
        print(f"\nStorage Info:")
        print(f"  Type: {storage_info['type']}")
        print(f"  Paper count: {storage_info['paper_count']}")

        # Search examples
        if storage_info["paper_count"] > 0:
            # Search by category
            ml_papers = json_storage.search_papers(
                categories=["Machine Learning in Finance"], limit=5
            )
            print(f"  ML papers found: {len(ml_papers)}")

            # Get all categories
            categories = json_storage.get_categories()
            print(
                f"  Categories: {', '.join(categories[:5])}{'...' if len(categories) > 5 else ''}"
            )

            # Get all tags
            tags = json_storage.get_tags()
            print(
                f"  Tags: {', '.join(tags[:10])}{'...' if len(tags) > 10 else ''}"
            )

    except Exception as e:
        logger.error(f"Storage operations failed: {e}")

    # 7. Show execution history
    history = agent.get_execution_history()
    if history:
        print(f"\nExecution History: {len(history)} pipeline runs")
        for execution in history[-3:]:  # Show last 3
            print(f"  {execution['pipeline_name']}: {execution['status']}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("Check ./data/quantmind_example/ for stored papers")
    print("=" * 80)


if __name__ == "__main__":
    main()
