#!/usr/bin/env python3
"""Demo pipeline showing how to create and use a custom flow."""

from pathlib import Path

# Import the custom flow
from flows.greeting_flow.flow import GreetingFlow
from quantmind.config.settings import load_config


def main():
    """Run the custom flow demo."""
    print("=== Custom Flow Demo: Greeting Flow ===")
    print()

    # Step 1: Load configuration from YAML using QuantMind settings
    config_path = Path(__file__).parent / "config.yaml"
    settings = load_config(config_path)

    # Step 2: Get the greeting flow configuration and convert to GreetingFlowConfig
    config = settings.flows["greeting_flow"]

    print(f"📁 Flow name: {config.name}")
    print(f"📁 Templates path: {config.prompt_templates_path}")

    print(f"✓ Loaded {len(config.prompt_templates)} prompt templates")
    print(f"✓ Configured {len(config.llm_blocks)} LLM blocks")
    print()

    # Step 3: Initialize the flow
    try:
        flow = GreetingFlow(config)
        print("✓ Flow initialized successfully")
    except Exception as e:
        print(f"✗ Flow initialization failed: {e}")
        print("This is expected without proper API configuration")
        return

    # Step 4: Run the flow with sample data
    user_inputs = [
        {"user_name": "Alice", "topic": "quantitative finance"},
        {"user_name": "Bob", "topic": "machine learning"},
        {"user_name": "Carol", "topic": "data science"},
    ]

    for user_input in user_inputs:
        print(f"\n--- Processing: {user_input} ---")
        try:
            result = flow.run(user_input)
            print("Greeting:", result.get("greeting", "N/A"))
            print("Suggestions:", result.get("suggestions", "N/A"))
        except Exception as e:
            print(f"✗ Flow execution failed: {e}")
            print("This is expected without proper API keys")

    print("\n=== Demo Complete ===")
    print("\nKey takeaways from this example:")
    print("• Simple flow configuration with dataclass")
    print("• YAML-based prompt templates")
    print("• Direct LLM block access (no wrapper methods)")
    print("• Python-based orchestration logic")
    print("• Easy to customize and extend")


if __name__ == "__main__":
    main()
