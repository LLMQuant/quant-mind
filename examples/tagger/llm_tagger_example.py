"""Example: LLM tagging for research papers."""

import os

from quantmind.config import LLMTaggerConfig
from quantmind.config.llm import LLMConfig
from quantmind.models.paper import Paper
from quantmind.tagger.llm_tagger import LLMTagger


def main():
    """Demonstrate simple LLM tagging."""
    # Create sample paper
    paper = Paper(
        title="LSTM Networks for High-Frequency Bitcoin Trading",
        abstract="""This study implements Long Short-Term Memory (LSTM) neural networks
        for predicting Bitcoin price movements in high-frequency trading scenarios.
        We use order book data and sentiment analysis from Twitter to train our model,
        achieving a Sharpe ratio of 1.8 over a 6-month period.""",
        authors=["Alice Johnson", "Bob Chen"],
        url="https://example.com/bitcoin-lstm-paper.pdf",
    )

    # Basic usage with defaults - Method 1: Using the convenient create() method
    print("=== Basic Usage (using create method) ===")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    tagger = LLMTagger(
        config=LLMTaggerConfig.create(
            model="gpt-4o-mini",
            api_key=api_key,
        )
    )

    tagged_paper = tagger.tag_paper(paper)

    print(f"Paper: {tagged_paper.title}")
    print(f"Generated Tags: {tagged_paper.tags}")
    print(f"Metadata: {tagged_paper.meta_info}")

    # Method 2: Using explicit LLMConfig composition
    print("\n=== Alternative Configuration (explicit LLMConfig) ===")
    if api_key:
        llm_config = LLMConfig(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.3,
        )

        tagger2 = LLMTagger(
            config=LLMTaggerConfig(
                llm_config=llm_config,
                max_tags=5,
            )
        )

        tagged_paper_alt = tagger2.tag_paper(paper)
        print(f"Paper: {tagged_paper_alt.title}")
        print(f"Generated Tags: {tagged_paper_alt.tags}")

    # Custom configuration with user instructions
    print("\n=== Custom Configuration (with user instructions) ===")
    if api_key:  # Only run if API key is available
        custom_tagger = LLMTagger(
            config=LLMTaggerConfig.create(
                model="gpt-4o-mini",
                api_key=api_key,
                custom_instructions="Use - to connect tags, like deep-learning.",
                max_tags=3,
                temperature=0.1,
            )
        )

        # Create another paper
        paper2 = Paper(
            title="Portfolio Optimization Using Reinforcement Learning",
            abstract="We apply deep Q-learning to portfolio allocation in equity markets.",
            authors=["Carol Smith"],
        )

        tagged_paper2 = custom_tagger.tag_paper(paper2)
        print(f"Paper: {tagged_paper2.title}")
        print(f"Generated Tags: {tagged_paper2.tags}")
    else:
        print("OPENAI_API_KEY not found, skipping custom configuration example")

    # Extract tags from arbitrary text
    # print("\n=== Text Analysis ===")
    # text = "This paper discusses volatility modeling in forex markets using GARCH models."
    # tags = tagger.extract_tags(text, "Volatility Study")
    # print(f"Tags from text: {tags}")


if __name__ == "__main__":
    main()
