"""Simple example: LLM tagging for research papers."""

from quantmind.config import LLMTaggerConfig
from quantmind.models.paper import Paper
from quantmind.tagger.llm_tagger import LLMTagger
from quantmind.utils.env import get_openai_api_key


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

    # Basic usage with defaults
    print("=== Basic Usage ===")
    tagger = LLMTagger(
        config=LLMTaggerConfig(
            llm_type="openai",
            llm_name="gpt-4o-mini",
            api_key=get_openai_api_key(required=True),
        )
    )

    tagged_paper = tagger.tag_paper(paper)

    print(f"Paper: {tagged_paper.title}")
    print(f"Generated Tags: {tagged_paper.tags}")
    print(f"Metadata: {tagged_paper.meta_info}")

    # Custom configuration
    print("\n=== Custom Configuration (with user instructions) ===")
    custom_tagger = LLMTagger(
        config=LLMTaggerConfig(
            llm_type="openai",
            llm_name="gpt-4o-mini",
            api_key=get_openai_api_key(required=False),
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

    # Extract tags from arbitrary text
    # print("\n=== Text Analysis ===")
    # text = "This paper discusses volatility modeling in forex markets using GARCH models."
    # tags = tagger.extract_tags(text, "Volatility Study")
    # print(f"Tags from text: {tags}")


if __name__ == "__main__":
    main()
