#!/usr/bin/env python3
"""Production-level Quant Paper Agent Pipeline using Gemini 2.5 Pro & Flash.

This pipeline demonstrates:
1. Real-world paper extraction from ArXiv
2. Advanced PDF parsing with LlamaParser
3. Intelligent summarization using Gemini models
4. QA generation for web engagement
5. Storage with rich metadata using meta_info approach
"""

import sys
from pathlib import Path
from typing import Optional

from flows.qa_flow.flow import QAFlow

from quantmind.config.settings import load_config
from quantmind.flow.summary_flow import SummaryFlow
from quantmind.models.paper import Paper
from quantmind.parsers.llama_parser import LlamaParser
from quantmind.sources.arxiv_source import ArxivSource
from quantmind.storage.local_storage import LocalStorage
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


# Color codes for better terminal output
class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # Emoji-like symbols
    ROCKET = "🚀"
    CHECK = "✓"
    CROSS = "❌"
    ARROW = "📡"
    DOWNLOAD = "📥"
    SEARCH = "🔍"
    WRITE = "📝"
    QUESTION = "❓"
    SAVE = "💾"
    CELEBRATE = "🎉"
    INFO = "📄"
    GEAR = "🔧"
    STOP = "⏹️"
    BULB = "💡"


def cprint(text: str, color: str = Colors.ENDC, bold: bool = False) -> None:
    """Print colored text to terminal."""
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.ENDC}")


def print_header(text: str) -> None:
    """Print a styled header."""
    cprint(f"\n{Colors.ROCKET} {text}", Colors.HEADER, bold=True)


def print_success(text: str) -> None:
    """Print a success message."""
    cprint(f"{Colors.CHECK} {text}", Colors.OKGREEN)


def print_error(text: str) -> None:
    """Print an error message."""
    cprint(f"{Colors.CROSS} {text}", Colors.FAIL, bold=True)


def print_info(text: str) -> None:
    """Print an info message."""
    cprint(f"{Colors.INFO} {text}", Colors.OKBLUE)


def print_step(icon: str, text: str) -> None:
    """Print a pipeline step."""
    cprint(f"\n{icon} {text}", Colors.OKCYAN, bold=True)


def get_unique_paper(
    arxiv_source: ArxivSource, local_storage: LocalStorage, retry_count: int = 3
) -> Optional[Paper]:
    """Get a unique paper from ArXiv focusing on quantitative finance."""
    search_queries = [
        "Large Language Models quantitative finance",
        "machine learning portfolio optimization",
        "deep learning algorithmic trading",
        "neural networks risk management finance",
        "AI financial markets",
    ]

    for query in search_queries:
        logger.info(f"Searching with query: {query}")

        for attempt in range(retry_count):
            try:
                papers = arxiv_source.search(
                    query=query,
                    max_results=3,
                )

                if not papers:
                    logger.warning(f"No papers found for query: {query}")
                    continue

                for paper in papers:
                    paper_id = paper.get_primary_id()
                    if paper_id not in local_storage._raw_files_index:
                        logger.info(f"Found unique paper: {paper.title}")
                        return paper
                    else:
                        logger.debug(f"Paper {paper_id} already processed")

            except Exception as e:
                logger.error(f"Error searching (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    logger.error(
                        f"Failed to search after {retry_count} attempts"
                    )

    logger.warning("No unique papers found across all queries")
    return None


def display_results(paper: Paper) -> None:
    """Display pipeline results in a colorful, user-friendly format."""
    # Header
    cprint("\n" + "=" * 80, Colors.HEADER)
    cprint("🎯 QUANT PAPER AGENT PIPELINE RESULTS", Colors.HEADER, bold=True)
    cprint("=" * 80, Colors.HEADER)

    # Paper Information
    print_info("PAPER INFORMATION:")
    cprint(f"Title: {paper.title}", Colors.OKBLUE, bold=True)
    cprint(f"Authors: {', '.join(paper.authors)}", Colors.OKBLUE)
    cprint(f"ArXiv ID: {paper.arxiv_id}", Colors.OKBLUE)
    cprint(f"Categories: {', '.join(paper.categories)}", Colors.OKBLUE)
    cprint(f"Published: {paper.published_date}", Colors.OKBLUE)

    # Display summary
    summary = paper.meta_info.get("summary")
    if summary:
        print_step("📋", "INTELLIGENT SUMMARY:")
        cprint("-" * 40, Colors.OKCYAN)
        # Truncate summary for display if too long
        display_summary = (
            summary[:500] + "..." if len(summary) > 500 else summary
        )
        cprint(display_summary, Colors.ENDC)

    # Display QA questions
    qa_data = paper.meta_info.get("qa_data")
    if qa_data:
        questions = qa_data.get("questions", [])
        print_step("❓", f"THOUGHTFUL QUESTIONS ({len(questions)} generated):")
        cprint("-" * 40, Colors.OKCYAN)

        for i, q_data in enumerate(questions, 1):
            question = q_data.get("question", "")
            category = q_data.get("category", "general")
            difficulty = q_data.get("difficulty", "intermediate")

            # Color code by difficulty
            difficulty_color = {
                "basic": Colors.OKGREEN,
                "intermediate": Colors.WARNING,
                "advanced": Colors.FAIL,
            }.get(difficulty.lower(), Colors.ENDC)

            cprint(f"\n{i}. {question}", Colors.ENDC, bold=True)
            print(
                f"   {Colors.ENDC}Category: {Colors.OKCYAN}{category.title()}{Colors.ENDC} | Difficulty: {difficulty_color}{difficulty.title()}{Colors.ENDC}"
            )

    # Display metadata
    print_step("🔧", "PROCESSING METADATA:")
    cprint("-" * 40, Colors.OKCYAN)
    if qa_data:
        cprint(f"QA Model: {qa_data.get('model_used', 'Unknown')}", Colors.ENDC)
        cprint(
            f"Generated: {qa_data.get('generated_at', 'Unknown')}", Colors.ENDC
        )
        cprint(
            f"Question Count: {qa_data.get('question_count', 0)}", Colors.ENDC
        )
        cprint(
            f"Focus Areas: {', '.join(qa_data.get('focus_areas', []))}",
            Colors.ENDC,
        )

    storage_path = paper.meta_info.get("storage_path", "Not stored")
    cprint(f"\nStorage Path: {storage_path}", Colors.OKGREEN)
    cprint("=" * 80, Colors.HEADER)


def main():
    """Execute the production quant paper agent pipeline."""
    print_header("Starting Production Quant Paper Agent Pipeline")
    cprint(
        "Using Gemini 2.5 Pro & Flash for advanced AI processing\n",
        Colors.OKCYAN,
    )

    try:
        # Step 1: Load configuration
        config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        logger.info("Loading configuration...")
        settings = load_config(config_path)

        # Step 2: Initialize components
        logger.info("Initializing pipeline components...")
        local_storage = LocalStorage(settings.storage)
        arxiv_source = ArxivSource(settings.source)
        llama_parser = LlamaParser(settings.parser)

        # Initialize flows
        summary_flow = SummaryFlow(settings.flows["summary_flow"])
        qa_flow = QAFlow(settings.flows["qa_flow"])

        print_success("All components initialized successfully")

        # Step 3: Get unique paper
        print_step("📡", "Searching for unique quantitative finance paper...")
        paper = get_unique_paper(arxiv_source, local_storage)
        if not paper:
            print_error(
                "No unique papers found. Try again later or check your search criteria."
            )
            return

        print_success(f"Found paper: {paper.title[:60]}...")

        # Step 4: Download and store raw content
        print_step("📥", "Downloading paper content...")
        local_storage.process_knowledge(paper)
        print_success("Paper downloaded successfully")

        # Step 5: Parse with LlamaParser
        print_step("🔍", "Parsing PDF content with LlamaParser...")
        paper = llama_parser.parse_paper(paper)

        if not paper.has_content():
            print_error("Failed to extract content from PDF")
            return

        print_success(f"Content extracted ({len(paper.content)} characters)")

        # Step 6: Generate summary using Gemini 2.5
        print_step("📝", "Generating intelligent summary with Gemini 2.5...")
        summary_result = summary_flow.run(paper)

        # Store summary in meta_info
        paper.meta_info["summary"] = summary_result
        paper.meta_info["summary_generated_at"] = paper.processed_at.isoformat()

        print_success("Summary generated successfully")

        # Step 7: Generate QA questions
        print_step("❓", "Generating thoughtful questions with Gemini 2.5...")
        qa_result = qa_flow.run(paper)
        # QA results are automatically stored in paper.meta_info by the flow

        print_success(f"Generated {len(qa_result)} thoughtful questions")

        # Step 8: Store final enriched paper
        print_step("💾", "Storing enriched paper with metadata...")
        local_storage.store_knowledge(paper)

        # Add storage path to meta_info
        storage_path = local_storage.get_knowledge_path(paper.get_primary_id())
        paper.meta_info["storage_path"] = str(storage_path)

        print_success("Paper stored with all metadata")

        # Step 9: Display results
        display_results(paper)

        cprint(
            f"\n🎉 Pipeline completed successfully!", Colors.OKGREEN, bold=True
        )
        cprint(f"Paper stored at: {storage_path}", Colors.OKGREEN)

    except KeyboardInterrupt:
        cprint("\n⏹️  Pipeline interrupted by user", Colors.WARNING, bold=True)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print_error(f"Pipeline failed: {e}")
        cprint("💡 Make sure you have:", Colors.OKCYAN, bold=True)
        cprint("   • Set GOOGLE_API_KEY environment variable", Colors.OKCYAN)
        cprint(
            "   • Set LLAMA_CLOUD_API_KEY environment variable", Colors.OKCYAN
        )
        cprint("   • Valid internet connection", Colors.OKCYAN)
        sys.exit(1)


if __name__ == "__main__":
    main()
