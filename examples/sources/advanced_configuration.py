"""Advanced ArXiv source configuration examples.

This example demonstrates advanced configuration options for the ArxivSource:
- Custom download directories
- Category filtering
- Rate limiting
- Content filtering
- Configuration validation
"""

import yaml
from pathlib import Path
from tempfile import TemporaryDirectory

from quantmind.config.sources import ArxivSourceConfig
from quantmind.sources.arxiv_source import ArxivSource
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


def finance_focused_config_example():
    """Demonstrate finance-focused configuration."""
    print("=== Finance-Focused Configuration ===")

    config = ArxivSourceConfig(
        # API settings
        max_results=20,
        sort_by="submittedDate",
        sort_order="descending",
        # Content filtering for finance
        include_categories=[
            "q-fin.ST",  # Statistical Finance
            "q-fin.TR",  # Trading and Market Microstructure
            "q-fin.PM",  # Portfolio Management
            "q-fin.RM",  # Risk Management
            "q-fin.CP",  # Computational Finance
        ],
        min_abstract_length=150,  # Longer abstracts for quality
        # Rate limiting to be respectful
        requests_per_second=0.8,
        timeout=45,
        # Download settings
        download_pdfs=True,
    )

    print("Configuration created:")
    print(f"- Max results: {config.max_results}")
    print(f"- Include categories: {config.include_categories}")
    print(f"- Min abstract length: {config.min_abstract_length}")
    print(f"- Requests per second: {config.requests_per_second}")
    print(f"- Download PDFs: {config.download_pdfs}")

    return config


def ai_research_config_example():
    """Demonstrate AI research configuration."""
    print("\n=== AI Research Configuration ===")

    config = ArxivSourceConfig(
        # API settings optimized for AI research
        max_results=50,
        sort_by="relevance",
        sort_order="descending",
        # AI/ML categories
        include_categories=[
            "cs.AI",  # Artificial Intelligence
            "cs.LG",  # Machine Learning
            "cs.CV",  # Computer Vision
            "cs.CL",  # Computation and Language
            "cs.NE",  # Neural and Evolutionary Computing
            "stat.ML",  # Machine Learning (Statistics)
        ],
        # Exclude some categories we're not interested in
        exclude_categories=[
            "cs.CR",  # Cryptography and Security
            "cs.SE",  # Software Engineering
        ],
        # Quality filters
        min_abstract_length=100,
        # Higher rate for research use
        requests_per_second=1.5,
        # No downloads for this config
        download_pdfs=False,
    )

    print("AI Research configuration:")
    print(f"- Focus areas: {len(config.include_categories)} categories")
    print(f"- Excluded: {config.exclude_categories}")
    print(f"- Sort by: {config.sort_by}")

    return config


def production_config_example():
    """Demonstrate production-ready configuration."""
    print("\n=== Production Configuration ===")

    with TemporaryDirectory() as temp_dir:
        download_dir = Path(temp_dir) / "arxiv_papers"

        config = ArxivSourceConfig(
            # Conservative settings for production
            max_results=100,
            timeout=60,
            retry_attempts=3,
            requests_per_second=0.5,  # Very conservative
            # Content quality controls
            min_abstract_length=200,
            # Download setup
            download_pdfs=True,
            download_dir=download_dir,
            # Broad categories for general research
            include_categories=[
                "cs.AI",
                "cs.LG",
                "stat.ML",  # AI/ML
                "q-fin.ST",
                "q-fin.TR",
                "q-fin.PM",  # Finance
                "math.OC",
                "math.PR",
                "math.ST",  # Math
            ],
        )

        print("Production configuration:")
        print(f"- Download directory: {config.download_dir}")
        print(f"- Timeout: {config.timeout}s")
        print(f"- Retry attempts: {config.retry_attempts}")
        print(f"- Rate limit: {config.requests_per_second} req/s")

        return config


def config_from_yaml_example():
    """Demonstrate loading configuration from YAML."""
    print("\n=== Configuration from YAML ===")

    yaml_config = """
    max_results: 25
    sort_by: "submittedDate"
    sort_order: "descending"

    # Download settings
    download_pdfs: true

    # Quality filters
    min_abstract_length: 120

    # Categories of interest
    include_categories:
      - "q-fin.ST"
      - "q-fin.TR"
      - "cs.AI"
      - "stat.ML"

    # Rate limiting
    requests_per_second: 1.0
    timeout: 30
    """

    # Parse YAML
    yaml_data = yaml.safe_load(yaml_config)

    # Create config from dictionary
    config = ArxivSourceConfig(**yaml_data)

    print("Configuration loaded from YAML:")
    print(f"- Max results: {config.max_results}")
    print(f"- Categories: {len(config.include_categories)}")
    print(f"- Download PDFs: {config.download_pdfs}")

    return config


def test_configurations():
    """Test different configurations with actual searches."""
    print("\n=== Testing Configurations ===")

    # Test finance config
    finance_config = finance_focused_config_example()
    finance_source = ArxivSource(config=finance_config)

    print("\nTesting finance configuration:")
    if finance_source.validate_config():
        papers = finance_source.search("portfolio optimization", max_results=3)
        print(f"✓ Found {len(papers)} finance papers")
        for paper in papers:
            print(f"  - {paper.title[:50]}...")
    else:
        print("✗ Finance configuration invalid")

    # Test AI config
    ai_config = ai_research_config_example()
    ai_source = ArxivSource(config=ai_config)

    print("\nTesting AI configuration:")
    if ai_source.validate_config():
        papers = ai_source.search("transformer neural network", max_results=3)
        print(f"✓ Found {len(papers)} AI papers")
        for paper in papers:
            print(f"  - {paper.title[:50]}...")
    else:
        print("✗ AI configuration invalid")


def config_validation_example():
    """Demonstrate configuration validation."""
    print("\n=== Configuration Validation ===")

    # Valid configuration
    try:
        valid_config = ArxivSourceConfig(
            max_results=10, sort_by="relevance", requests_per_second=1.0
        )
        print("✓ Valid configuration created successfully")
    except Exception as e:
        print(f"✗ Valid configuration failed: {e}")

    # Invalid configurations
    invalid_configs = [
        {"sort_by": "invalid_sort"},
        {"sort_order": "invalid_order"},
        {"max_results": -1},
        {"requests_per_second": 0},
    ]

    for i, invalid_config in enumerate(invalid_configs, 1):
        try:
            ArxivSourceConfig(**invalid_config)
            print(f"✗ Invalid config {i} should have failed!")
        except Exception as e:
            print(
                f"✓ Invalid config {i} correctly rejected: {type(e).__name__}"
            )


def compare_configurations():
    """Compare search results across different configurations."""
    print("\n=== Configuration Comparison ===")

    query = "machine learning finance"

    configs = {
        "Default": ArxivSourceConfig(),
        "Relevance": ArxivSourceConfig(sort_by="relevance"),
        "Recent": ArxivSourceConfig(sort_by="submittedDate"),
        "Finance-only": ArxivSourceConfig(
            include_categories=["q-fin.ST", "q-fin.TR"]
        ),
    }

    for name, config in configs.items():
        source = ArxivSource(config=config)
        papers = source.search(query, max_results=3)

        print(f"\n{name} configuration:")
        print(f"  Found {len(papers)} papers")
        if papers:
            print(f"  Top result: {papers[0].title[:60]}...")
            print(f"  Categories: {papers[0].categories}")


def main():
    """Run all configuration examples."""
    examples = [
        config_validation_example,
        config_from_yaml_example,
        test_configurations,
        compare_configurations,
    ]

    for i, example in enumerate(examples, 1):
        try:
            print(f"\n{'=' * 70}")
            print(f"Configuration Example {i}/{len(examples)}")
            example()
        except Exception as e:
            logger.error(f"Error in configuration example {i}: {e}")

        # Small delay between examples
        if i < len(examples):
            import time

            time.sleep(0.5)

    print(f"\n{'=' * 70}")
    print("All configuration examples completed!")


if __name__ == "__main__":
    main()
