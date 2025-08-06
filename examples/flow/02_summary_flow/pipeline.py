#!/usr/bin/env python3
"""Demo pipeline for SummaryFlow with custom chunking strategy."""

from pathlib import Path

from mock_data import get_custom_chunking_example

from quantmind.config.settings import load_config
from quantmind.flow.summary_flow import SummaryFlow


def main():
    """Run the summary flow demo with custom chunking."""
    print("=== Summary Flow Demo: Custom Chunking ===")
    print()

    # Step 1: Load configuration from YAML using QuantMind settings
    config_path = Path(__file__).parent / "config.yaml"
    settings = load_config(config_path)

    # Step 2: Get the summary flow configuration
    config = settings.flows["summary_demo"]

    print(f"📁 Flow name: {config.name}")
    print(f"📁 Templates path: {config.prompt_templates_path}")
    print(f"✓ Loaded {len(config.prompt_templates)} prompt templates")
    print(f"✓ Configured {len(config.llm_blocks)} LLM blocks")
    print()

    # Step 3: Get sample paper with custom chunking strategy
    paper, custom_chunker = get_custom_chunking_example()

    print(f"📄 Paper: '{paper.title}' ({len(paper.content)} chars)")
    print(f"🧩 Custom chunker: {custom_chunker.__name__}")
    print()

    # Step 4: Update config to use custom chunking
    from quantmind.config.flows import ChunkingStrategy

    config.chunk_strategy = ChunkingStrategy.BY_CUSTOM
    config.chunk_custom_strategy = custom_chunker

    print(f"⚙️  Chunking: {config.use_chunking}")
    print(f"⚙️  Strategy: {config.chunk_strategy.value}")
    print(f"⚙️  Custom function: {config.chunk_custom_strategy.__name__}")
    print()

    # Step 5: Initialize and run the flow
    try:
        flow = SummaryFlow(config)
        print("✓ Flow initialized successfully")

        # Show LLM blocks configuration
        print("\n🤖 LLM Configuration:")
        for name, llm_config in config.llm_blocks.items():
            key_status = "✓ Found" if llm_config.api_key else "✗ Missing"
            print(f"  • {name}: {llm_config.model} ({key_status})")
        print()

        # Run the flow
        print("🚀 Running summary flow...")
        result = flow.run(paper)

        print(f"✓ Generated summary ({len(result)} chars)")
        print("\n" + "=" * 50)
        print("📝 SUMMARY:")
        print("=" * 50)
        print(result)
        print("=" * 50)

    except Exception as e:
        print(f"✗ Flow execution failed: {e}")
        print("💡 Make sure you have set up API keys in .env file")

    print("\n=== Demo Complete ===")
    print("\nKey features demonstrated:")
    print("• Custom chunking strategy implementation")
    print("• Two-stage summarization (cheap + powerful LLMs)")
    print("• Automatic API key resolution from environment")
    print("• YAML-based prompt templates")
    print("• Type-safe configuration loading")


if __name__ == "__main__":
    main()
