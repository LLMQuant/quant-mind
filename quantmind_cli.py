#!/usr/bin/env python3
"""QuantMind Command Line Interface.

Main entry point for the QuantMind knowledge extraction system.
Provides commands for running pipelines, managing components, and configuration.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from quantmind.workflow.agent import WorkflowAgent
from quantmind.sources.arxiv_source import ArxivSource
from quantmind.parsers.pdf_parser import PDFParser
from quantmind.tagger.rule_tagger import RuleTagger
from quantmind.tagger.llm_tagger import LLMTagger
from quantmind.storage.json_storage import JSONStorage
from quantmind.config.settings import (
    load_config,
    create_default_config,
    save_config,
)
from quantmind.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def setup_agent_from_config(config_path: Optional[str] = None) -> WorkflowAgent:
    """Set up WorkflowAgent from configuration.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configured WorkflowAgent
    """
    # Load configuration
    if config_path and Path(config_path).exists():
        settings = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        settings = create_default_config()
        logger.info("Using default configuration")

    # Create agent
    agent = WorkflowAgent(config=settings.workflow.__dict__)

    # Register components
    component_registry = {
        "ArxivSource": ArxivSource,
        "PDFParser": PDFParser,
        "RuleTagger": RuleTagger,
        "LLMTagger": LLMTagger,
        "JSONStorage": JSONStorage,
    }

    # Register sources
    for name, source_config in settings.get_enabled_sources().items():
        if source_config.type in component_registry:
            source = component_registry[source_config.type](
                config=source_config.config
            )
            agent.register_source(name, source)
            logger.debug(f"Registered source: {name}")

    # Register parsers
    for name, parser_config in settings.get_enabled_parsers().items():
        if parser_config.type in component_registry:
            parser = component_registry[parser_config.type](
                config=parser_config.config
            )
            agent.register_parser(name, parser)
            logger.debug(f"Registered parser: {name}")

    # Register taggers
    for name, tagger_config in settings.get_enabled_taggers().items():
        if tagger_config.type in component_registry:
            tagger = component_registry[tagger_config.type](
                config=tagger_config.config
            )
            agent.register_tagger(name, tagger)
            logger.debug(f"Registered tagger: {name}")

    # Register storages
    for name, storage_config in settings.get_enabled_storages().items():
        if storage_config.type in component_registry:
            storage = component_registry[storage_config.type](
                config=storage_config.config
            )
            agent.register_storage(name, storage)
            logger.debug(f"Registered storage: {name}")

    return agent


def cmd_extract(args) -> None:
    """Run paper extraction."""
    logger.info(f"Starting paper extraction: {args.query}")

    agent = setup_agent_from_config(args.config)

    try:
        papers = agent.run_quick_extraction(
            source_name=args.source,
            query=args.query,
            max_papers=args.max_papers,
            tagger_name=args.tagger,
        )

        print(f"\nExtracted {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"{i:2d}. {paper.title}")
            print(
                f"     Authors: {', '.join(paper.authors[:2])}{'...' if len(paper.authors) > 2 else ''}"
            )
            print(f"     Categories: {', '.join(paper.categories)}")
            if paper.arxiv_id:
                print(f"     ArXiv: {paper.arxiv_id}")
            print()

        logger.info("Extraction completed successfully")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


def cmd_pipeline(args) -> None:
    """Run a full extraction pipeline."""
    logger.info(f"Creating pipeline: {args.name}")

    agent = setup_agent_from_config(args.config)

    try:
        pipeline = agent.create_extraction_pipeline(
            name=args.name,
            source_name=args.source,
            query=args.query,
            max_papers=args.max_papers,
            parser_name=args.parser,
            tagger_name=args.tagger,
            storage_name=args.storage,
        )

        logger.info("Executing pipeline...")
        results = agent.execute_pipeline(args.name)

        # Display results
        stats = agent.get_pipeline_status(args.name)
        print(f"\nPipeline '{args.name}' completed:")
        print(f"  Status: {stats['status']}")
        print(f"  Total tasks: {stats['total_tasks']}")
        print(f"  Duration: {stats.get('duration', 'N/A')} seconds")
        print(f"  Results: {len(results)} task outputs")

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


def cmd_search(args) -> None:
    """Search stored papers."""
    logger.info("Searching stored papers")

    agent = setup_agent_from_config(args.config)

    if not agent.storages:
        logger.error("No storage backends configured")
        sys.exit(1)

    storage_name = list(agent.storages.keys())[0]
    storage = agent.storages[storage_name]

    try:
        papers = storage.search_papers(
            query=args.query,
            categories=args.categories.split(",") if args.categories else None,
            tags=args.tags.split(",") if args.tags else None,
            limit=args.limit,
        )

        print(f"\nFound {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"{i:2d}. {paper.title}")
            print(f"     Categories: {', '.join(paper.categories)}")
            print(
                f"     Tags: {', '.join(paper.tags[:3])}{'...' if len(paper.tags) > 3 else ''}"
            )
            print()

    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)


def cmd_status(args) -> None:
    """Show system status."""
    logger.info("Checking system status")

    agent = setup_agent_from_config(args.config)

    print("QuantMind System Status")
    print("=" * 40)
    print(f"Sources: {len(agent.sources)}")
    for name in agent.sources:
        print(f"  - {name}")

    print(f"Parsers: {len(agent.parsers)}")
    for name in agent.parsers:
        print(f"  - {name}")

    print(f"Taggers: {len(agent.taggers)}")
    for name in agent.taggers:
        print(f"  - {name}")

    print(f"Storages: {len(agent.storages)}")
    for name in agent.storages:
        print(f"  - {name}")

    print(f"\nConfiguration:")
    print(f"  Max workers: {agent.max_workers}")
    print(f"  Retry attempts: {agent.retry_attempts}")
    print(f"  Timeout: {agent.timeout}s")
    print(f"  Deduplication: {agent.enable_deduplication}")

    # Storage stats
    if agent.storages:
        storage = list(agent.storages.values())[0]
        info = storage.get_storage_info()
        print(f"\nStorage Statistics:")
        print(f"  Papers stored: {info['paper_count']}")
        if info["paper_count"] > 0:
            categories = storage.get_categories()
            print(f"  Categories: {len(categories)}")
            tags = storage.get_tags()
            print(f"  Tags: {len(tags)}")


def cmd_config(args) -> None:
    """Manage configuration."""
    if args.action == "create":
        logger.info(f"Creating default configuration: {args.output}")
        settings = create_default_config()
        save_config(settings, args.output)
        print(f"Default configuration saved to {args.output}")

    elif args.action == "show":
        if args.config and Path(args.config).exists():
            settings = load_config(args.config)
            print(f"Configuration from {args.config}:")
        else:
            settings = create_default_config()
            print("Default configuration:")

        print(f"  Log level: {settings.log_level}")
        print(f"  Data directory: {settings.data_dir}")
        print(f"  Max workers: {settings.workflow.max_workers}")
        print(
            f"  Sources: {len(settings.sources)} ({len(settings.get_enabled_sources())} enabled)"
        )
        print(
            f"  Parsers: {len(settings.parsers)} ({len(settings.get_enabled_parsers())} enabled)"
        )
        print(
            f"  Taggers: {len(settings.taggers)} ({len(settings.get_enabled_taggers())} enabled)"
        )
        print(
            f"  Storages: {len(settings.storages)} ({len(settings.get_enabled_storages())} enabled)"
        )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QuantMind - Intelligent Knowledge Extraction for Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s extract "machine learning finance" --max-papers 10
  %(prog)s pipeline ml_finance_pipeline "cat:q-fin.ST" --storage json
  %(prog)s search --categories "Machine Learning in Finance" --limit 5
  %(prog)s status
  %(prog)s config create --output config.yaml
        """,
    )

    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Enable quiet mode (errors only)",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract papers from sources"
    )
    extract_parser.add_argument("query", help="Search query")
    extract_parser.add_argument(
        "--source", default="arxiv", help="Source name (default: arxiv)"
    )
    extract_parser.add_argument(
        "--tagger", default="rule", help="Tagger name (default: rule)"
    )
    extract_parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="Maximum papers (default: 10)",
    )
    extract_parser.set_defaults(func=cmd_extract)

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run full extraction pipeline"
    )
    pipeline_parser.add_argument("name", help="Pipeline name")
    pipeline_parser.add_argument("query", help="Search query")
    pipeline_parser.add_argument(
        "--source", default="arxiv", help="Source name (default: arxiv)"
    )
    pipeline_parser.add_argument("--parser", help="Parser name")
    pipeline_parser.add_argument(
        "--tagger", default="rule", help="Tagger name (default: rule)"
    )
    pipeline_parser.add_argument(
        "--storage", default="json", help="Storage name (default: json)"
    )
    pipeline_parser.add_argument(
        "--max-papers",
        type=int,
        default=50,
        help="Maximum papers (default: 50)",
    )
    pipeline_parser.set_defaults(func=cmd_pipeline)

    # Search command
    search_parser = subparsers.add_parser("search", help="Search stored papers")
    search_parser.add_argument("--query", help="Text query")
    search_parser.add_argument(
        "--categories", help="Categories (comma-separated)"
    )
    search_parser.add_argument("--tags", help="Tags (comma-separated)")
    search_parser.add_argument(
        "--limit", type=int, default=20, help="Maximum results (default: 20)"
    )
    search_parser.set_defaults(func=cmd_search)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.set_defaults(func=cmd_status)

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(
        dest="action", help="Config actions"
    )

    create_config_parser = config_subparsers.add_parser(
        "create", help="Create default config"
    )
    create_config_parser.add_argument(
        "--output", default="quantmind_config.yaml", help="Output file"
    )

    show_config_parser = config_subparsers.add_parser(
        "show", help="Show current config"
    )

    config_parser.set_defaults(func=cmd_config)

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    setup_logger(level=log_level)

    # Run command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
