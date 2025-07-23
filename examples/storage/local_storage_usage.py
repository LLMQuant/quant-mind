"""Local storage usage example for QuantMind.

This example demonstrates:
1. Storing raw files from content bytes directly
2. Using process_knowledge for Paper objects
3. Automatic PDF downloading for Paper objects
"""

from datetime import datetime, timezone
from pathlib import Path

from quantmind.config import LocalStorageConfig
from quantmind.models import Paper
from quantmind.storage import LocalStorage


def demonstrate_enhanced_raw_file_storage():
    """Demonstrate storing raw files from content."""
    print("=== Enhanced Raw File Storage Demo ===")

    # Initialize storage
    config = LocalStorageConfig(storage_dir=Path("./demo_data"))
    storage = LocalStorage(config)

    # Example 1: Store content directly as bytes
    print("\n1. Storing content directly as bytes:")

    # Simulate PDF content downloaded from ArXiv
    pdf_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj"

    # Store PDF content directly
    pdf_path = storage.store_raw_file(
        file_id="paper_1", content=pdf_content, file_extension=".pdf"
    )
    print(f"  Stored PDF content to: {pdf_path}")

    # Store text content
    text_content = "# Research Paper Abstract\n\nSample content".encode("utf-8")

    text_path = storage.store_raw_file(
        file_id="paper_1_abstract", content=text_content, file_extension=".md"
    )
    print(f"  Stored text content to: {text_path}")

    return storage


def demonstrate_paper_specialized_storage():
    """Demonstrate Paper-specific storage with automatic handling."""
    print("\n=== Paper Specialized Storage Demo ===")

    # Initialize storage
    config = LocalStorageConfig(storage_dir=Path("./demo_data"))
    storage = LocalStorage(config)

    # Paper with PDF URL
    paper_with_pdf = Paper(
        title="Machine Learning in Quantitative Finance",
        abstract="This paper explores ML techniques in quantitative finance.",
        authors=["John Smith", "Jane Doe"],
        arxiv_id="2024.0001",
        pdf_url="https://arxiv.org/pdf/2024.0001.pdf",
        categories=["q-fin.CP", "cs.LG"],
        published_date=datetime.now(timezone.utc),
        source="arxiv",
    )

    # Store with specialized handling
    paper_id = storage.process_knowledge(paper_with_pdf)
    print(f"  Stored paper with ID: {paper_id}")

    return storage


def main():
    """Main demonstration function."""
    print("🚀 QuantMind Enhanced Storage Demonstration")

    try:
        demonstrate_enhanced_raw_file_storage()
        demonstrate_paper_specialized_storage()
        print(f"\n🎉 All demonstrations completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
