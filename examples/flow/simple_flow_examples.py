"""Simple examples demonstrating the flow framework in QuantMind."""

import os

from quantmind.flow import QAFlow, SummaryFlow
from quantmind.config import QAFlowConfig, SummaryFlowConfig
from quantmind.models.paper import Paper
from quantmind.models.content import KnowledgeItem
from datetime import datetime


def create_sample_knowledge_item() -> KnowledgeItem:
    """Create a sample knowledge item for demonstration."""
    return KnowledgeItem(
        title="Deep Learning for Algorithmic Trading",
        abstract=(
            "This research explores deep learning applications in algorithmic trading. "
            "We analyze LSTM and Transformer models for financial time series prediction, "
            "achieving Sharpe ratios exceeding 2.5 in high-frequency trading scenarios. "
            "Our approach uses neural networks for feature extraction and risk management."
        ),
        authors=["John Smith", "Jane Doe"],
        source="arXiv",
        content_type="research_paper",
        published_date=datetime(2023, 10, 15),
        content=(
            "# Deep Learning for Algorithmic Trading\n\n"
            "## Introduction\n"
            "Algorithmic trading has evolved with machine learning advances. "
            "This research presents a comprehensive analysis of deep learning models "
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
        categories=["machine_learning", "algorithmic_trading", "finance"],
        tags=[
            "LSTM",
            "transformers",
            "trading",
            "neural_networks",
            "risk_management",
        ],
    )


def create_sample_paper() -> Paper:
    """Create a sample paper for backward compatibility demonstration."""
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


def example_qa_flow():
    """Demonstrate Q&A flow usage with enhanced prompt engineering."""
    print("=== Q&A Flow Example ===")

    # Create configuration using the new LLMConfig composition pattern
    config = QAFlowConfig.create(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        # Flow-specific parameters
        num_questions=3,
        include_different_difficulties=True,
        # Optional: Custom prompt template using {{variable}} syntax
        prompt_template=(
            "{{system_prompt}}\n\n"
            "Generate {{num_questions}} insightful Q&A pairs for this content:\n\n"
            "Title: {{title}}\n"
            "Abstract: {{abstract}}\n"
            "Content Type: {{content_type}}\n"
            "Categories: {{categories}}\n"
            "Tags: {{tags}}\n\n"
            "Focus Areas: {{question_categories}}\n"
            "Difficulty Levels: {{difficulty_levels}}\n\n"
            "{{custom_instructions}}\n\n"
            "Return your response in structured JSON format."
        ),
        # Custom instructions are now stored in LLMConfig
        custom_instructions="Focus on practical trading applications and implementation details",
    )

    # Initialize flow
    qa_flow = QAFlow(config)

    # Create sample knowledge item
    knowledge_item = create_sample_knowledge_item()

    print(f"Generating Q&A for: {knowledge_item.title}")
    print(f"Content Type: {knowledge_item.content_type}")

    # Generate Q&A pairs
    qa_pairs = qa_flow.execute(knowledge_item)

    # Display results
    print(f"\nGenerated {len(qa_pairs)} Q&A pairs:")
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\nQ{i} ({qa.difficulty_level}, {qa.category}):")
        print(f"  {qa.question}")
        print(f"A{i}: {qa.answer}")

    # Generate summary
    if qa_pairs:
        summary = qa_flow.generate_summary(qa_pairs)
        print(f"\nSummary: {summary}")
    else:
        print("\nNo Q&A pairs generated (check API key configuration)")


def example_summary_flow():
    """Demonstrate summary flow usage with template variables."""
    print("\n=== Summary Flow Example ===")

    # Create configuration using the new LLMConfig composition pattern
    config = SummaryFlowConfig.create(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        # Flow-specific parameters
        summary_type="comprehensive",
        max_summary_length=600,
        include_key_findings=True,
        include_methodology=True,
        include_results=True,
        include_implications=True,
        focus_on_quantitative_aspects=True,
        highlight_innovations=True,
        # Custom template variables
        template_variables={
            "analysis_focus": "trading strategies and risk management",
            "target_audience": "quantitative researchers and practitioners",
        },
    )

    # Initialize flow
    summary_flow = SummaryFlow(config)

    # Create sample knowledge item
    knowledge_item = create_sample_knowledge_item()

    print(f"Generating summary for: {knowledge_item.title}")

    # Generate summary
    summary_result = summary_flow.execute(knowledge_item)

    # Display results
    if summary_result.get("error"):
        print(f"\nError: {summary_result['error_message']}")
    else:
        print(f"\nSummary generated successfully!")
        print(f"Content Type: {summary_result.get('content_type', 'Unknown')}")
        print(f"Summary Type: {summary_result['summary_type']}")
        print(f"Word Count: {summary_result['word_count']}")
        print(f"\nSummary:")
        print("-" * 40)
        print(summary_result["summary"])


def example_flow_comparison():
    """Compare flow approaches and demonstrate template system."""
    print("\n=== Flow Comparison & Template Demo ===")

    knowledge_item = create_sample_knowledge_item()

    print(
        "1. Q&A Flow - Generates questions and answers for knowledge extraction"
    )
    qa_config = QAFlowConfig.create(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        num_questions=2,
        # Show template variables extraction
        template_variables={"special_focus": "implementation challenges"},
    )
    qa_flow = QAFlow(qa_config)

    # Demonstrate template variable preview
    template_vars = qa_flow.get_template_variables(knowledge_item)
    print(
        f"   Template variables available: {len(template_vars)} (title, abstract, content, etc.)"
    )

    qa_pairs = qa_flow.execute(knowledge_item)
    print(f"   Generated {len(qa_pairs)} Q&A pairs")

    print("\n2. Summary Flow - Generates comprehensive content summaries")
    summary_config = SummaryFlowConfig.create(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        summary_type="brief",
        max_summary_length=300,
    )
    summary_flow = SummaryFlow(summary_config)
    summary_result = summary_flow.execute(knowledge_item)
    if not summary_result.get("error"):
        print(f"   Generated {summary_result['word_count']} word summary")
    else:
        print("   Summary generation failed")


def example_backward_compatibility():
    """Demonstrate backward compatibility with Paper objects."""
    print("\n=== Backward Compatibility Example ===")

    # Show that flows can still work with Paper objects (since Paper inherits from KnowledgeItem)
    paper = create_sample_paper()

    print(f"Using Paper object: {paper.title}")
    print(
        f"Paper inherits from KnowledgeItem: {isinstance(paper, KnowledgeItem)}"
    )

    # Use with QA flow
    qa_config = QAFlowConfig.create(
        model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), num_questions=1
    )
    qa_flow = QAFlow(qa_config)

    try:
        qa_pairs = qa_flow.execute(
            paper
        )  # Paper works because it's a KnowledgeItem
        print(f"   Generated {len(qa_pairs)} Q&A pairs from Paper object")
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    """Run flow examples."""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Warning: OPENAI_API_KEY not set. Examples will not call actual LLM."
        )
        print("Set your API key: export OPENAI_API_KEY=your_key_here")
        print("Examples will still demonstrate the flow structure.\n")

    print("QuantMind Flow Framework Examples")
    print("=" * 45)

    try:
        # Run examples
        example_qa_flow()
        example_summary_flow()
        example_flow_comparison()
        example_backward_compatibility()

        print("\n" + "=" * 45)
        print("All flow examples completed!")
        print("\nKey Features Demonstrated:")
        print("- Enhanced prompt engineering with {{variable}} templates")
        print("- Generic KnowledgeItem support (not just Paper)")
        print("- Backward compatibility with existing Paper objects")
        print("- Custom template variables and instructions")
        print("- Template variable preview and debugging")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This is expected if no API key is configured.")
