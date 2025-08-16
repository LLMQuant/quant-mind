#!/usr/bin/env python3
"""Demo pipeline showing how to create and use a podcast flow."""

import json
from pathlib import Path

# Import the custom flow
from flows.podcast_flow.flow import PodcastFlow
from quantmind.config.settings import load_config


def main():
    """Run the podcast flow demo."""
    print("=== Custom Flow Demo: Podcast Flow ===")
    print()

    # Step 1: Load configuration from YAML using QuantMind settings
    config_path = Path(__file__).parent / "config.yaml"
    settings = load_config(config_path)

    # Step 2: Get the podcast flow configuration and convert to PodcastFlowConfig
    config = settings.flows["podcast_flow"]

    print(f"📁 Flow name: {config.name}")
    print(f"📁 Templates path: {config.prompt_templates_path}")

    print(f"✓ Loaded {len(config.prompt_templates)} prompt templates")
    print(f"✓ Configured {len(config.llm_blocks)} LLM blocks")
    print()

    # Step 3: Initialize the flow
    try:
        flow = PodcastFlow(config)
        print("✓ Flow initialized successfully")
    except Exception as e:
        print(f"✗ Flow initialization failed: {e}")
        print("This is expected without proper API configuration")
        return

    # Step 4: Run the flow with sample data
    sample_inputs = [
        {
            "summary": "Artificial Intelligence is transforming healthcare in unprecedented ways. Machine learning algorithms can now diagnose diseases with accuracy rates exceeding human doctors. AI-powered imaging systems detect early-stage cancers that might be missed by traditional methods.",
        }
    ]

    for i, input_data in enumerate(sample_inputs, 1):
        print(f"\n--- Processing Podcast {i}: {input_data['summary']} ---")
        try:
            result = flow.run(summary=input_data["summary"])
            assert isinstance(
                result, dict
            ), "Flow output should be a dictionary"
            with open(f"podcast_script_{i}.json", "w") as f:
                json.dump(result, f, indent=4)
            print(f"✓ Podcast script saved to podcast_script_{i}.json")
        except Exception as e:
            print(f"✗ Flow execution failed: {e}")
            print("This is expected without proper API keys")

    print("\n=== Demo Complete ===")
    print("\nKey takeaways from this example:")
    print("• Podcast flow configuration with dataclass")
    print("• YAML-based prompt templates for main")
    print("• Multiple LLM blocks for different content types")
    print("• Python-based orchestration logic")
    print("• Easy to customize and extend for different podcast styles")


if __name__ == "__main__":
    main()
