#!/usr/bin/env python3
"""Configuration example for QuantMind.

This example demonstrates how to use the configuration system to set up
QuantMind components and workflows.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import quantmind
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantmind.config.settings import (
    Settings,
    create_default_config,
    save_config,
    load_config,
)
from quantmind.workflow.agent import WorkflowAgent
from quantmind.sources.arxiv_source import ArxivSource
from quantmind.parsers.pdf_parser import PDFParser
from quantmind.tagger.rule_tagger import RuleTagger
from quantmind.tagger.llm_tagger import LLMTagger
from quantmind.storage.json_storage import JSONStorage
from quantmind.utils.logger import setup_logger, get_logger

# Set up logging
setup_logger()
logger = get_logger(__name__)


def create_sample_config():
    """Create a sample configuration for QuantMind."""
    # Start with default configuration
    settings = create_default_config()

    # Customize workflow settings
    settings.workflow.max_workers = 6
    settings.workflow.retry_attempts = 3
    settings.workflow.timeout = 600  # 10 minutes
    settings.workflow.enable_deduplication = True
    settings.workflow.quality_threshold = 0.6

    # Configure sources
    settings.sources["arxiv"].config.update(
        {"max_results": 200, "rate_limit_delay": 1.0}
    )

    # Add additional source for news
    from quantmind.config.settings import SourceConfig

    settings.sources["financial_news"] = SourceConfig(
        name="financial_news",
        type="NewsSource",
        config={
            "api_key": "${NEWS_API_KEY}",
            "sources": ["reuters", "bloomberg", "financial-times"],
            "max_articles": 100,
        },
        enabled=False,  # Disabled by default
    )

    # Configure parsers
    settings.parsers["pdf"].config.update(
        {
            "method": "marker",  # Use AI-powered parsing
            "download_pdfs": True,
            "max_file_size": 100,
            "cache_dir": "./cache/pdfs",
        }
    )

    # Add web parser
    from quantmind.config.settings import ParserConfig

    settings.parsers["web"] = ParserConfig(
        name="web",
        type="WebParser",
        config={
            "user_agent": "QuantMind/1.0",
            "timeout": 30,
            "max_content_length": 1000000,
        },
    )

    # Configure taggers
    settings.taggers["rule"].config.update(
        {
            "case_sensitive": False,
            "custom_categories": {
                "ESG Finance": ["esg", "sustainable finance", "green bonds"],
                "Crypto Finance": [
                    "cryptocurrency",
                    "bitcoin",
                    "blockchain",
                    "defi",
                ],
                "High Frequency Trading": [
                    "hft",
                    "high frequency",
                    "microsecond",
                    "latency",
                ],
            },
        }
    )

    # Enable LLM tagger with custom configuration
    settings.taggers["llm"].enabled = True
    settings.taggers["llm"].config.update(
        {
            "model_type": "openai",
            "model_name": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 1500,
            "custom_prompt_template": "financial_classification",
        }
    )

    # Configure storage
    settings.storages["json"].config.update(
        {
            "storage_dir": "./data/quantmind",
            "auto_backup": True,
            "max_backup_count": 10,
            "compression": True,
        }
    )

    # Add database storage option
    from quantmind.config.settings import StorageConfig

    settings.storages["database"] = StorageConfig(
        name="database",
        type="DatabaseStorage",
        config={
            "connection_string": "${DATABASE_URL}",
            "table_prefix": "quantmind_",
            "enable_full_text_search": True,
            "connection_pool_size": 5,
        },
        enabled=False,  # Disabled by default
    )

    # Set global settings
    settings.log_level = "INFO"
    settings.arxiv_max_results = 500

    return settings


def demonstrate_config_usage():
    """Demonstrate how to use configuration in practice."""
    logger.info("Creating sample configuration...")

    # Create configuration
    settings = create_sample_config()

    # Save configuration to file
    config_path = Path("./examples/quantmind/sample_config.yaml")
    save_config(settings, config_path)
    logger.info(f"Saved configuration to {config_path}")

    # Load configuration from file
    logger.info("Loading configuration from file...")
    loaded_settings = load_config(config_path)

    # Create WorkflowAgent from configuration
    agent = WorkflowAgent(config=loaded_settings.workflow.__dict__)

    # Register components based on configuration
    logger.info("Registering components from configuration...")

    # Register enabled sources
    for name, source_config in loaded_settings.get_enabled_sources().items():
        if source_config.type == "ArxivSource":
            source = ArxivSource(config=source_config.config)
            agent.register_source(name, source)
            logger.info(f"Registered source: {name}")

    # Register enabled parsers
    for name, parser_config in loaded_settings.get_enabled_parsers().items():
        if parser_config.type == "PDFParser":
            parser = PDFParser(config=parser_config.config)
            agent.register_parser(name, parser)
            logger.info(f"Registered parser: {name}")

    # Register enabled taggers
    for name, tagger_config in loaded_settings.get_enabled_taggers().items():
        if tagger_config.type == "RuleTagger":
            tagger = RuleTagger(config=tagger_config.config)
            agent.register_tagger(name, tagger)
            logger.info(f"Registered tagger: {name}")
        elif tagger_config.type == "LLMTagger" and os.getenv("OPENAI_API_KEY"):
            tagger = LLMTagger(config=tagger_config.config)
            agent.register_tagger(name, tagger)
            logger.info(f"Registered tagger: {name}")

    # Register enabled storages
    for name, storage_config in loaded_settings.get_enabled_storages().items():
        if storage_config.type == "JSONStorage":
            storage = JSONStorage(config=storage_config.config)
            agent.register_storage(name, storage)
            logger.info(f"Registered storage: {name}")

    # Display agent status
    print("\n" + "=" * 60)
    print("AGENT CONFIGURATION")
    print("=" * 60)
    print(f"Sources: {list(agent.sources.keys())}")
    print(f"Parsers: {list(agent.parsers.keys())}")
    print(f"Taggers: {list(agent.taggers.keys())}")
    print(f"Storages: {list(agent.storages.keys())}")
    print(f"Max workers: {agent.max_workers}")
    print(f"Retry attempts: {agent.retry_attempts}")
    print(f"Timeout: {agent.timeout}")

    return agent, loaded_settings


def show_configuration_details(settings):
    """Show detailed configuration information."""
    print("\n" + "=" * 60)
    print("CONFIGURATION DETAILS")
    print("=" * 60)

    print(f"\nWorkflow Settings:")
    print(f"  Max workers: {settings.workflow.max_workers}")
    print(f"  Retry attempts: {settings.workflow.retry_attempts}")
    print(f"  Timeout: {settings.workflow.timeout}")
    print(f"  Deduplication: {settings.workflow.enable_deduplication}")
    print(f"  Quality threshold: {settings.workflow.quality_threshold}")

    print(f"\nGlobal Settings:")
    print(f"  Log level: {settings.log_level}")
    print(f"  Storage directory: {settings.storage.storage_dir}")
    print(f"  ArXiv max results: {settings.arxiv_max_results}")

    print(f"\nSources ({len(settings.sources)}):")
    for name, source in settings.sources.items():
        status = "✓" if source.enabled else "✗"
        print(f"  {status} {name} ({source.type})")

    print(f"\nParsers ({len(settings.parsers)}):")
    for name, parser in settings.parsers.items():
        status = "✓" if parser.enabled else "✗"
        print(f"  {status} {name} ({parser.type})")

    print(f"\nTaggers ({len(settings.taggers)}):")
    for name, tagger in settings.taggers.items():
        status = "✓" if tagger.enabled else "✗"
        print(f"  {status} {name} ({tagger.type})")

    print(f"\nStorages ({len(settings.storages)}):")
    for name, storage in settings.storages.items():
        status = "✓" if storage.enabled else "✗"
        print(f"  {status} {name} ({storage.type})")


def main():
    """Run the configuration example."""
    logger.info("Starting QuantMind configuration example")

    try:
        # Create and demonstrate configuration
        agent, settings = demonstrate_config_usage()

        # Show configuration details
        show_configuration_details(settings)

        # Test a simple extraction if we have components
        if agent.sources and agent.taggers:
            logger.info("\nTesting configured pipeline...")

            source_name = list(agent.sources.keys())[0]
            tagger_name = list(agent.taggers.keys())[0]

            try:
                papers = agent.run_quick_extraction(
                    source_name=source_name,
                    query="machine learning finance",
                    max_papers=3,
                    tagger_name=tagger_name,
                )

                print(
                    f"\nTest extraction successful: {len(papers)} papers processed"
                )

            except Exception as e:
                logger.warning(f"Test extraction failed: {e}")

        print("\n" + "=" * 60)
        print("Configuration example completed successfully!")
        print("Check sample_config.yaml for the generated configuration")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Configuration example failed: {e}")
        raise


if __name__ == "__main__":
    main()
