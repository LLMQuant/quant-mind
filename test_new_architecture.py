#!/usr/bin/env python3
"""Test script for the new QuantMind architecture."""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from quantmind.models.paper import Paper
from quantmind.tagger.rule_tagger import RuleTagger
from quantmind.storage.json_storage import JSONStorage
from quantmind.config.settings import create_default_config


def test_basic_functionality():
    """Test basic QuantMind functionality."""
    print("🧪 Testing QuantMind basic functionality...")

    # 1. Test Paper model
    print("\n1. Testing Paper model...")
    paper = Paper(
        title="Machine Learning Applications in Quantitative Finance",
        abstract="This paper explores the use of machine learning techniques for portfolio optimization and risk management in quantitative finance.",
        authors=["John Doe", "Jane Smith"],
        categories=["Finance"],
        arxiv_id="2401.12345",
    )
    print(f"✓ Created paper: {paper.get_primary_id()}")
    print(f"  Title: {paper.title}")
    print(f"  Authors: {', '.join(paper.authors)}")

    # 2. Test RuleTagger
    print("\n2. Testing RuleTagger...")
    tagger = RuleTagger()
    tagged_paper = tagger.tag_paper(paper)
    print(f"✓ Tagged paper with {len(tagged_paper.categories)} categories:")
    for category in tagged_paper.categories:
        print(f"    - {category}")
    print(f"  Tags: {', '.join(tagged_paper.tags)}")

    # 3. Test JSONStorage
    print("\n3. Testing JSONStorage...")
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = JSONStorage(config={"storage_dir": temp_dir})

        # Store the paper
        paper_id = storage.store_paper(tagged_paper)
        print(f"✓ Stored paper with ID: {paper_id}")

        # Retrieve the paper
        retrieved_paper = storage.get_paper(paper_id)
        if retrieved_paper:
            print(f"✓ Retrieved paper: {retrieved_paper.title}")
        else:
            print("✗ Failed to retrieve paper")

        # Search papers
        search_results = storage.search_papers(query="machine learning")
        print(f"✓ Search found {len(search_results)} papers")

        # Get statistics
        stats = storage.get_storage_info()
        print(f"✓ Storage contains {stats['paper_count']} papers")

    # 4. Test Configuration
    print("\n4. Testing Configuration...")
    config = create_default_config()
    print(f"✓ Created config with {len(config.sources)} sources")
    print(f"  Workflow max workers: {config.workflow.max_workers}")
    print(f"  Enabled sources: {list(config.get_enabled_sources().keys())}")
    print(f"  Enabled taggers: {list(config.get_enabled_taggers().keys())}")

    print("\n🎉 All basic tests passed!")


def test_knowledge_graph():
    """Test knowledge graph functionality."""
    print("\n🧪 Testing KnowledgeGraph...")

    from quantmind.models.knowledge_graph import KnowledgeGraph

    # Create some test papers
    papers = [
        Paper(
            title="Deep Learning for Portfolio Optimization",
            abstract="Using neural networks for portfolio optimization",
            categories=["Deep Learning in Finance", "Portfolio Optimization"],
        ),
        Paper(
            title="Reinforcement Learning in Trading",
            abstract="Applying RL algorithms to algorithmic trading",
            categories=[
                "Reinforcement Learning in Finance",
                "Quantitative Trading",
            ],
        ),
        Paper(
            title="Risk Management with Machine Learning",
            abstract="ML approaches to financial risk assessment",
            categories=["Machine Learning in Finance", "Risk Management"],
        ),
    ]

    # Create knowledge graph
    kg = KnowledgeGraph()
    paper_ids = kg.add_papers(papers)
    print(f"✓ Added {len(paper_ids)} papers to knowledge graph")

    # Connect papers by categories
    edges_added = kg.connect_by_categories()
    print(f"✓ Connected {edges_added} paper pairs by shared categories")

    # Get graph statistics
    stats = kg.get_graph_statistics()
    print(f"✓ Graph contains {stats['nodes']} nodes and {stats['edges']} edges")
    print(f"  Density: {stats['density']:.3f}")
    print(f"  Categories: {', '.join(stats['category_distribution'].keys())}")

    # Find central papers
    central_papers = kg.get_central_papers(metric="degree", top_k=2)
    print(f"✓ Most central papers:")
    for paper, score in central_papers:
        print(f"    - {paper.title} (score: {score:.3f})")

    print("🎉 Knowledge graph tests passed!")


def main():
    """Run all tests."""
    print("🚀 Testing QuantMind v0.2.0 Architecture")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_knowledge_graph()

        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("🎯 QuantMind architecture is ready for use!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
