"""Basic ArXiv source usage example.

This example demonstrates how to use the ArxivSource class for:
- Basic paper search
- Retrieving papers by ID
- Getting recent papers
- Configuring the source with different settings
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from quantmind.config.sources import ArxivSourceConfig
from quantmind.sources.arxiv_source import ArxivSource
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


def basic_search_example():
    """Demonstrate basic search functionality."""
    print("=== Basic Search Example ===")

    # Create source with default configuration
    source = ArxivSource()

    # Search for machine learning papers
    papers = source.search("machine learning", max_results=5)

    print(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.title}")
        print(
            f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}"
        )
        print(f"   Categories: {', '.join(paper.categories)}")
        print(f"   ArXiv ID: {paper.arxiv_id}")
        print()


def configured_search_example():
    """Demonstrate search with custom configuration."""
    print("=== Configured Search Example ===")

    # Create configuration for finance-focused search
    config = ArxivSourceConfig(
        max_results=10,
        sort_by="relevance",
        sort_order="descending",
        include_categories=["q-fin.ST", "q-fin.TR", "q-fin.PM"],
        min_abstract_length=100,
        requests_per_second=0.5,  # Slower to be respectful
    )

    source = ArxivSource(config=config)

    # Search for quantitative finance papers
    papers = source.search("portfolio optimization", max_results=3)

    print(f"Found {len(papers)} finance papers:")
    for paper in papers:
        print(f"- {paper.title}")
        print(f"  Categories: {', '.join(paper.categories)}")
        print(f"  Abstract length: {len(paper.abstract)} chars")
        print()


def get_by_id_example():
    """Demonstrate retrieving papers by ID."""
    print("=== Get by ID Example ===")

    source = ArxivSource()

    # Try to get a specific paper by arXiv ID
    paper_ids = [
        "2301.12345",
        "1706.03762",
    ]  # Second one is "Attention Is All You Need"

    for paper_id in paper_ids:
        paper = source.get_by_id(paper_id)
        if paper:
            print(f"Found paper: {paper.title}")
            print(f"Authors: {', '.join(paper.authors)}")
            print(f"Published: {paper.published_date}")
            print()
        else:
            print(f"Paper {paper_id} not found")


def recent_papers_example():
    """Demonstrate getting recent papers."""
    print("=== Recent Papers Example ===")

    source = ArxivSource()

    # Get recent AI papers from the last 3 days
    papers = source.get_by_timeframe(days=3, categories=["cs.AI", "cs.LG"])

    print(f"Found {len(papers)} recent AI/ML papers:")
    for paper in papers[:5]:  # Show first 5
        print(f"- {paper.title}")
        print(f"  Published: {paper.published_date}")
        print()


def download_example():
    """Demonstrate PDF download functionality."""
    print("=== PDF Download Example ===")

    with TemporaryDirectory() as temp_dir:
        # Configure source with PDF downloads enabled
        config = ArxivSourceConfig(
            download_pdfs=True,
            download_dir=Path(temp_dir),
            max_results=2,
            requests_per_second=0.5,
            proxies={
                "http": "http://127.0.0.1:7890",
                "https": "http://127.0.0.1:7890",
                "all_proxy": "socks5://127.0.0.1:7890",
            },
        )

        source = ArxivSource(config=config)

        # Search for a few papers
        papers = source.search("attention mechanism", max_results=2)

        if papers:
            print(f"Downloading PDFs for {len(papers)} papers...")

            # Download PDFs
            download_paths = source.download_papers_pdfs(papers)

            for paper, path in zip(papers, download_paths):
                if path:
                    print(f"✓ Downloaded: {paper.title}")
                    print(f"  File: {path.name}")
                    print(f"  Size: {path.stat().st_size} bytes")
                else:
                    print(f"✗ Failed to download: {paper.title}")
                print()
        else:
            print("No papers found for download example")


def batch_retrieval_example():
    """Demonstrate batch retrieval of papers."""
    print("=== Batch Retrieval Example ===")

    source = ArxivSource()

    # List of paper IDs to retrieve
    paper_ids = [
        "1706.03762",  # Attention Is All You Need
        "1512.03385",  # ResNet
        "1409.1556",  # GAN
        "nonexistent",  # This one won't be found
    ]

    papers = source.get_batch(paper_ids)

    print(f"Requested {len(paper_ids)} papers, found {len(papers)}:")
    for paper in papers:
        print(f"- {paper.title}")
        print(f"  ArXiv ID: {paper.arxiv_id}")
        print()


def category_search_example():
    """Demonstrate category-specific search."""
    print("=== Category Search Example ===")

    source = ArxivSource()

    # Search in specific categories
    categories = ["q-fin.ST", "cs.AI"]

    for category in categories:
        papers = source.search_by_category(category, max_results=3)
        print(f"Category {category}: {len(papers)} papers")

        for paper in papers:
            print(f"  - {paper.title[:60]}...")
        print()


def main():
    """Run all examples."""
    examples = [
        # basic_search_example,
        # configured_search_example,
        # get_by_id_example,
        # recent_papers_example,
        # batch_retrieval_example,
        # category_search_example,
        download_example,  # This one creates files, so run it last
    ]

    for i, example in enumerate(examples, 1):
        try:
            print(f"\n{'=' * 60}")
            print(f"Example {i}/{len(examples)}")
            example()
        except Exception as e:
            logger.error(f"Error in example {i}: {e}")

        # Add a small delay between examples to be respectful to arXiv
        if i < len(examples):
            import time

            time.sleep(1)

    print(f"\n{'=' * 60}")
    print("All examples completed!")


if __name__ == "__main__":
    main()
