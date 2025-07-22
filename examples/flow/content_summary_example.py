"""Example usage of SummaryFlow for generating content summaries."""

import os
from quantmind.models.paper import Paper
from quantmind.models.content import KnowledgeItem
from quantmind.flow import SummaryFlow
from quantmind.config import SummaryFlowConfig
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


def create_sample_knowledge_item() -> KnowledgeItem:
    """Create a sample knowledge item for demonstration."""
    return KnowledgeItem(
        title="Deep Reinforcement Learning for Portfolio Optimization",
        authors=["John Smith", "Jane Doe", "Bob Johnson"],
        abstract="We propose a novel deep reinforcement learning approach for portfolio optimization that leverages attention mechanisms and multi-agent systems to achieve superior risk-adjusted returns. Our method combines traditional financial theory with modern machine learning techniques to address the challenges of high-dimensional, non-stationary financial markets.",
        categories=["q-fin.PM", "cs.AI", "q-fin.TR"],
        tags=[
            "portfolio optimization",
            "reinforcement learning",
            "deep learning",
            "attention mechanisms",
        ],
        content_type="research_paper",
        content="""
        Abstract: Portfolio optimization remains a fundamental challenge in quantitative finance,
        particularly in high-dimensional and non-stationary market environments. Traditional
        approaches often struggle with the complexity and dynamics of modern financial markets.

        Introduction: Deep reinforcement learning (DRL) has shown remarkable success in
        various domains, from game playing to robotics. In this research, we explore its
        application to portfolio optimization, a domain characterized by high uncertainty,
        complex dynamics, and multiple competing objectives.

        Methodology: Our approach employs a multi-agent reinforcement learning framework
        where each agent specializes in different aspects of portfolio management. We use
        attention mechanisms to capture temporal dependencies and cross-asset relationships.
        The agents learn through interaction with a realistic market simulator that incorporates
        transaction costs, market impact, and regulatory constraints.

        Results: We evaluate our method on historical data from major equity markets over
        a 10-year period. Our approach achieves a Sharpe ratio of 1.85, significantly
        outperforming traditional methods including Markowitz optimization (1.12) and
        equal-weight portfolios (0.89). The method also demonstrates robust performance
        across different market regimes and asset classes.

        Conclusion: The results demonstrate the potential of deep reinforcement learning
        for portfolio optimization, particularly in complex, high-dimensional market
        environments. Future work will explore extensions to other asset classes and
        the integration of alternative data sources.
        """,
        source="arxiv",
        url="https://arxiv.org/abs/2023.12345",
    )


def create_sample_paper() -> Paper:
    """Create a sample paper for backward compatibility demonstration."""
    return Paper(
        title="Deep Reinforcement Learning for Portfolio Optimization",
        authors=["John Smith", "Jane Doe", "Bob Johnson"],
        abstract="We propose a novel deep reinforcement learning approach for portfolio optimization that leverages attention mechanisms and multi-agent systems to achieve superior risk-adjusted returns. Our method combines traditional financial theory with modern machine learning techniques to address the challenges of high-dimensional, non-stationary financial markets.",
        categories=["q-fin.PM", "cs.AI", "q-fin.TR"],
        tags=[
            "portfolio optimization",
            "reinforcement learning",
            "deep learning",
            "attention mechanisms",
        ],
        full_text="""
        Abstract: Portfolio optimization remains a fundamental challenge in quantitative finance,
        particularly in high-dimensional and non-stationary market environments. Traditional
        approaches often struggle with the complexity and dynamics of modern financial markets.

        Introduction: Deep reinforcement learning (DRL) has shown remarkable success in
        various domains, from game playing to robotics. In this paper, we explore its
        application to portfolio optimization, a domain characterized by high uncertainty,
        complex dynamics, and multiple competing objectives.

        Methodology: Our approach employs a multi-agent reinforcement learning framework
        where each agent specializes in different aspects of portfolio management. We use
        attention mechanisms to capture temporal dependencies and cross-asset relationships.
        The agents learn through interaction with a realistic market simulator that incorporates
        transaction costs, market impact, and regulatory constraints.

        Results: We evaluate our method on historical data from major equity markets over
        a 10-year period. Our approach achieves a Sharpe ratio of 1.85, significantly
        outperforming traditional methods including Markowitz optimization (1.12) and
        equal-weight portfolios (0.89). The method also demonstrates robust performance
        across different market regimes and asset classes.

        Conclusion: The results demonstrate the potential of deep reinforcement learning
        for portfolio optimization, particularly in complex, high-dimensional market
        environments. Future work will explore extensions to other asset classes and
        the integration of alternative data sources.
        """,
        source="arxiv",
        url="https://arxiv.org/abs/2023.12345",
    )


def basic_summary_example():
    """Demonstrate basic content summary generation."""
    print("=== Basic Content Summary Example ===\n")

    # Create configuration using the new LLMConfig composition pattern
    config = SummaryFlowConfig.create(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        # Flow-specific parameters
        summary_type="comprehensive",
        max_summary_length=600,
        output_format="narrative",
        include_key_findings=True,
        include_methodology=True,
        include_results=True,
        include_implications=True,
        focus_on_quantitative_aspects=True,
        highlight_innovations=True,
        include_limitations=True,
    )

    # Initialize flow
    flow = SummaryFlow(config)

    # Create sample knowledge item
    knowledge_item = create_sample_knowledge_item()

    # Generate summary
    print(f"Content: {knowledge_item.title}")
    print(f"Type: {knowledge_item.content_type}")
    print(f"Authors: {', '.join(knowledge_item.authors)}")
    print(f"Abstract: {knowledge_item.abstract[:200]}...")
    print("\n" + "=" * 50 + "\n")

    try:
        result = flow.execute(knowledge_item)

        if result.get("error"):
            print(f"Error: {result['error_message']}")
        else:
            print("Generated Summary:")
            print("-" * 30)
            print(result["summary"])
            print(f"\nWord count: {result['word_count']}")
            print(f"Summary type: {result['summary_type']}")
            print(f"Content type: {result.get('content_type', 'Unknown')}")

    except Exception as e:
        print(f"Error generating summary: {e}")


def structured_summary_example():
    """Demonstrate structured summary generation with custom templates."""
    print("\n=== Structured Summary Example ===\n")

    # Create configuration for structured output with custom template using new pattern
    config = SummaryFlowConfig.create(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        # Flow-specific parameters
        summary_type="technical",
        max_summary_length=800,
        output_format="structured",
        include_key_findings=True,
        include_methodology=True,
        include_results=True,
        include_implications=True,
        focus_on_quantitative_aspects=True,
        highlight_innovations=True,
        include_limitations=True,
        # Custom template to demonstrate the template system
        prompt_template=(
            "{{system_prompt}}\n\n"
            "Analyze this {{content_type}} and create a {{summary_type}} summary:\n\n"
            "Title: {{title}}\n"
            "Authors: {{authors}}\n"
            "Abstract: {{abstract}}\n"
            "Categories: {{categories}}\n"
            "Tags: {{tags}}\n\n"
            "Content:\n{{content}}\n\n"
            "Generate a {{output_format}} summary (max {{max_summary_length}} words).\n"
            "{{custom_instructions}}"
        ),
    )

    # Initialize flow
    flow = SummaryFlow(config)

    # Create sample knowledge item
    knowledge_item = create_sample_knowledge_item()

    try:
        result = flow.execute(knowledge_item)

        if result.get("error"):
            print(f"Error: {result['error_message']}")
        else:
            print("Structured Summary Results:")
            print("-" * 30)

            if result.get("key_findings"):
                print(f"Key Findings: {result['key_findings']}")

            if result.get("methodology"):
                print(f"Methodology: {result['methodology']}")

            if result.get("results"):
                print(f"Results: {result['results']}")

            if result.get("implications"):
                print(f"Implications: {result['implications']}")

            if result.get("limitations"):
                print(f"Limitations: {result['limitations']}")

            print(f"\nContent ID: {result['content_id']}")
            print(f"Content Type: {result['content_type']}")
            print(f"Output format: {result['output_format']}")

    except Exception as e:
        print(f"Error generating structured summary: {e}")


def brief_summary_example():
    """Demonstrate brief summary generation."""
    print("\n=== Brief Summary Example ===\n")

    # Create configuration using new pattern
    config = SummaryFlowConfig.create(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        summary_type="brief",
        max_summary_length=200,
        output_format="narrative",
    )

    # Initialize flow
    flow = SummaryFlow(config)

    # Create sample knowledge item
    knowledge_item = create_sample_knowledge_item()

    try:
        # Use convenience method for brief summary
        brief_summary = flow.generate_brief_summary(knowledge_item)

        if brief_summary:
            print("Brief Summary:")
            print("-" * 20)
            print(brief_summary)
            print(f"\nLength: {len(brief_summary.split())} words")
        else:
            print("Failed to generate brief summary")

    except Exception as e:
        print(f"Error generating brief summary: {e}")


def executive_summary_example():
    """Demonstrate executive summary generation."""
    print("\n=== Executive Summary Example ===\n")

    # Create configuration using new pattern
    config = SummaryFlowConfig.create(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        summary_type="executive",
        max_summary_length=500,
        output_format="bullet_points",
        include_key_findings=True,
        include_methodology=False,  # Less technical for executives
        include_results=True,
        include_implications=True,
        focus_on_quantitative_aspects=False,  # More business-focused
        highlight_innovations=True,
        include_limitations=True,
    )

    # Initialize flow
    flow = SummaryFlow(config)

    # Create sample knowledge item
    knowledge_item = create_sample_knowledge_item()

    try:
        # Use convenience method for executive summary
        exec_summary = flow.generate_executive_summary(knowledge_item)

        if exec_summary.get("error"):
            print(f"Error: {exec_summary['error_message']}")
        else:
            print("Executive Summary:")
            print("-" * 20)
            print(exec_summary["summary"])

            if exec_summary.get("bullet_points"):
                print("\nKey Points:")
                for i, point in enumerate(exec_summary["bullet_points"][:5], 1):
                    print(f"{i}. {point}")

            print(f"\nSummary type: {exec_summary['summary_type']}")
            print(f"Output format: {exec_summary['output_format']}")
            print(
                f"Content type: {exec_summary.get('content_type', 'Unknown')}"
            )

    except Exception as e:
        print(f"Error generating executive summary: {e}")


def custom_instructions_example():
    """Demonstrate custom instructions and template variables."""
    print("\n=== Custom Instructions & Template Variables Example ===\n")

    # Create configuration with custom instructions and template variables using new pattern
    config = SummaryFlowConfig.create(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        summary_type="comprehensive",
        max_summary_length=600,
        output_format="narrative",
        # Custom instructions stored in LLMConfig
        custom_instructions="""
        Please focus on the practical trading implications of this research.
        Highlight any novel trading strategies or risk management approaches.
        Consider the scalability and implementation challenges in real-world trading systems.
        Emphasize the competitive advantages this approach might provide to quantitative trading firms.
        """,
        # Custom template variables to demonstrate the flexibility
        template_variables={
            "business_focus": "quantitative trading applications",
            "target_audience": "portfolio managers and quant researchers",
            "competitive_context": "hedge funds and proprietary trading firms",
        },
    )

    # Initialize flow
    flow = SummaryFlow(config)

    # Create sample knowledge item
    knowledge_item = create_sample_knowledge_item()

    try:
        result = flow.execute(knowledge_item)

        if result.get("error"):
            print(f"Error: {result['error_message']}")
        else:
            print("Custom Instructions Summary:")
            print("-" * 30)
            print(result["summary"])
            print(
                f"\nCustom instructions applied: {bool(config.llm_config.custom_instructions)}"
            )
            print(f"Template variables used: {len(config.template_variables)}")

    except Exception as e:
        print(f"Error generating custom summary: {e}")


def backward_compatibility_example():
    """Demonstrate backward compatibility with Paper objects."""
    print("\n=== Backward Compatibility Example ===\n")

    # Create configuration using new pattern
    config = SummaryFlowConfig.create(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        summary_type="brief",
        max_summary_length=300,
        output_format="narrative",
    )

    # Initialize flow
    flow = SummaryFlow(config)

    # Create sample paper (legacy object)
    paper = create_sample_paper()

    print(f"Using Paper object: {paper.title}")
    print(f"Paper is KnowledgeItem: {isinstance(paper, KnowledgeItem)}")

    try:
        # Paper objects work seamlessly since they inherit from KnowledgeItem
        result = flow.execute(paper)

        if result.get("error"):
            print(f"Error: {result['error_message']}")
        else:
            print("\nBackward compatibility successful!")
            print("Paper object processed by SummaryFlow:")
            print("-" * 40)
            print(result["summary"][:200] + "...")
            print(f"\nContent ID: {result['content_id']}")
            print(f"Content Type: {result.get('content_type', 'paper')}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all content summary flow examples."""
    print("Content Summary Flow Examples")
    print("=" * 50)

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Some examples may fail without a valid API key.")
        print()

    # Run examples
    basic_summary_example()
    structured_summary_example()
    brief_summary_example()
    executive_summary_example()
    custom_instructions_example()
    backward_compatibility_example()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nKey Flow Features Demonstrated:")
    print("- Enhanced prompt engineering with {{variable}} templates")
    print("- Generic KnowledgeItem support (not limited to papers)")
    print("- Custom template variables and instructions")
    print("- Backward compatibility with Paper objects")
    print("- Flexible summary types and output formats")
    print("- Template-based configuration system")


if __name__ == "__main__":
    main()
