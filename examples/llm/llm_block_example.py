"""Example demonstrating the LLMBlock architecture."""

import os

from dotenv import load_dotenv

from quantmind.config import LLMConfig
from quantmind.llm import create_llm_block

load_dotenv()


def example_basic_llm_block():
    """Example 1: Basic LLMBlock usage."""
    print("=== Example 1: Basic LLMBlock Usage ===")

    # Create LLM configuration
    config = LLMConfig(
        model="deepseek/deepseek-chat",  # LiteLLM format
        temperature=0.0,
        max_tokens=1000,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )

    # Create LLMBlock
    llm_block = create_llm_block(config)

    # Test connection
    if llm_block.test_connection():
        print("✅ LLMBlock connection successful!")

        # Generate text
        response = llm_block.generate_text("What is machine learning?")
        print(f"Response: {response[:100]}...")

        # Get block info
        print(f"Block info: {llm_block.get_info()}")
    else:
        print("❌ LLMBlock connection failed")


def example_advanced_features():
    """Example 5: Advanced features and configuration."""
    print("\n=== Example 5: Advanced Features ===")

    # Advanced configuration
    config = LLMConfig(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=2000,
        top_p=0.9,
        timeout=120,
        retry_attempts=5,
        retry_delay=2.0,
        system_prompt="You are a quantitative finance expert.",
        custom_instructions="Always provide practical examples and code snippets.",
        extra_params={"frequency_penalty": 0.1, "presence_penalty": 0.1},
    )

    print(f"Advanced config: {config.model_dump()}")

    # Create LLMBlock
    llm_block = create_llm_block(config)

    # Using context manager for temporary changes
    with llm_block.temporary_config(temperature=0.0, max_tokens=500):
        print("Inside temporary config context")
        print(f"Current block info: {llm_block.get_info()}")

    print("Outside temporary config context")
    print(f"Current block info: {llm_block.get_info()}")


if __name__ == "__main__":
    print("🚀 QuantMind LLMBlock Architecture Examples")
    print("=" * 50)

    # Run examples
    example_basic_llm_block()
    example_advanced_features()

    print("\n✅ All examples completed!")
