"""Simple examples demonstrating the two basic workflows in QuantMind."""

import os

from quantmind.workflow import QAWorkflow, AnalyzerWorkflow
from quantmind.config import QAWorkflowConfig, AnalyzerWorkflowConfig
from quantmind.models.paper import Paper
from datetime import datetime


def create_sample_paper() -> Paper:
    """Create a sample paper for demonstration."""
    return Paper(
        title="Deep Learning for Algorithmic Trading",
        abstract=(
            "This paper explores deep learning applications in algorithmic trading. "
            "We analyze LSTM and Transformer models for financial time series prediction, "
            "achieving Sharpe ratios exceeding 2.5 in high-frequency trading scenarios. "
            "Our approach uses neural networks for feature extraction and risk management."
        ),
        authors=["John Smith", "Jane Doe"],
        source="arXiv",
        arxiv_id="2023.12345",
        published_date=datetime(2023, 10, 15),
        full_text=(
            "# Deep Learning for Algorithmic Trading\n\n"
            "## Introduction\n"
            "Algorithmic trading has evolved with machine learning advances. "
            "This paper presents a comprehensive analysis of deep learning models "
            "for financial market prediction and trading strategy optimization.\n\n"
            "## Methodology\n"
            "We employed LSTM networks for temporal pattern recognition in "
            "high-frequency trading data. Our architecture includes attention "
            "mechanisms and regularization techniques to prevent overfitting.\n\n"
            "## Results\n"
            "The LSTM model achieved a Sharpe ratio of 2.8 with maximum drawdown "
            "below 5%. Risk-adjusted returns outperformed traditional strategies "
            "by 15% over the backtesting period.\n\n"
            "## Risk Management\n"
            "We implemented dynamic position sizing and stop-loss mechanisms "
            "based on volatility forecasts from the neural network model."
        ),
    )


def example_qa_workflow():
    """Demonstrate Q&A workflow usage."""
    print("=== Q&A Workflow Example ===")

    # Create configuration
    config = QAWorkflowConfig(
        llm_type="openai",
        llm_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        num_questions=3,
        include_different_difficulties=True,
        custom_instructions="Focus on practical trading applications",
    )

    # Initialize workflow
    qa_workflow = QAWorkflow(config)

    # Create sample paper
    paper = create_sample_paper()

    print(f"Generating Q&A for: {paper.title}")

    # Generate Q&A pairs
    qa_pairs = qa_workflow.execute(paper)

    # Display results
    print(f"\nGenerated {len(qa_pairs)} Q&A pairs:")
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\nQ{i} ({qa.difficulty_level}, {qa.category}):")
        print(f"  {qa.question}")
        print(f"A{i}: {qa.answer}")

    # Generate summary
    if qa_pairs:
        summary = qa_workflow.generate_summary(qa_pairs)
        print(f"\nSummary: {summary}")
    else:
        print("\nNo Q&A pairs generated (check API key configuration)")


def example_analyzer_workflow():
    """Demonstrate analyzer workflow usage."""
    print("\n=== Analyzer Workflow Example ===")

    # Create configuration
    config = AnalyzerWorkflowConfig(
        llm_type="openai",
        llm_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tags=6,
        tag_confidence_threshold=0.7,
        generate_methodology_summary=True,
        generate_results_summary=True,
    )

    # Initialize workflow
    analyzer_workflow = AnalyzerWorkflow(config)

    # Create sample paper
    paper = create_sample_paper()

    print(f"Analyzing paper: {paper.title}")

    # Analyze paper for tags
    primary_tags, secondary_tags = analyzer_workflow.execute(paper)

    # Display tag results
    print(f"\nPrimary Tags ({len(primary_tags)}):")
    for tag in primary_tags:
        print(f"  {tag}")

    print(f"\nSecondary Tags ({len(secondary_tags)}):")
    for tag in secondary_tags:
        print(f"  {tag}")

    # Generate summaries if enabled
    if config.generate_methodology_summary:
        methodology_summary = analyzer_workflow.generate_methodology_summary(
            paper
        )
        if methodology_summary:
            print(f"\nMethodology Summary:\n{methodology_summary}")

    if config.generate_results_summary:
        results_summary = analyzer_workflow.generate_results_summary(paper)
        if results_summary:
            print(f"\nResults Summary:\n{results_summary}")

    # Generate tag summary
    if primary_tags or secondary_tags:
        tag_summary = analyzer_workflow.generate_tag_summary(
            primary_tags, secondary_tags
        )
        print(f"\nTag Summary: {tag_summary}")
    else:
        print("\nNo tags generated (check API key configuration)")


def example_workflow_comparison():
    """Compare the two workflow approaches."""
    print("\n=== Workflow Comparison ===")

    paper = create_sample_paper()

    print(
        "1. Q&A Workflow - Generates questions and answers for knowledge extraction"
    )
    qa_config = QAWorkflowConfig(
        llm_type="openai", api_key=os.getenv("OPENAI_API_KEY"), num_questions=2
    )
    qa_workflow = QAWorkflow(qa_config)
    qa_pairs = qa_workflow.execute(paper)
    print(f"   Generated {len(qa_pairs)} Q&A pairs")

    print("\n2. Analyzer Workflow - Extracts structured tags and summaries")
    analyzer_config = AnalyzerWorkflowConfig(
        llm_type="openai", api_key=os.getenv("OPENAI_API_KEY"), max_tags=4
    )
    analyzer_workflow = AnalyzerWorkflow(analyzer_config)
    primary_tags, secondary_tags = analyzer_workflow.execute(paper)
    print(
        f"   Generated {len(primary_tags)} primary and {len(secondary_tags)} secondary tags"
    )


if __name__ == "__main__":
    """Run workflow examples."""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Warning: OPENAI_API_KEY not set. Examples will not call actual LLM."
        )
        print("Set your API key: export OPENAI_API_KEY=your_key_here")
        print("Examples will still demonstrate the workflow structure.\n")

    print("QuantMind Workflow Examples")
    print("=" * 40)

    try:
        # Run examples
        example_qa_workflow()
        example_analyzer_workflow()
        example_workflow_comparison()

        print("\n" + "=" * 40)
        print("All workflow examples completed!")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This is expected if no API key is configured.")
