#!/usr/bin/env python3
"""Demo pipeline showing how to create and use a podcast flow."""

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
            "intro": "AI revolutionizing healthcare with machine learning and predictive analytics",
            "outro": "Future of AI in medicine and ethical considerations",
        },
        {
            "summary": "Quantum computing is revolutionizing cryptography and drug discovery. Quantum algorithms can solve complex problems that classical computers cannot handle. This technology has the potential to break current encryption methods while enabling new breakthroughs in pharmaceutical research.",
            "intro": "Quantum computing breakthroughs in cryptography and drug discovery",
            "outro": "Implications for cybersecurity and pharmaceutical research",
        },
        {
            "summary": "Climate change is accelerating faster than predicted. Rising global temperatures are causing extreme weather events, sea level rise, and ecosystem disruptions. Renewable energy adoption is increasing but not fast enough to meet climate targets.",
            "intro": "Climate change acceleration and its global impacts",
            "outro": "Renewable energy adoption and climate action urgency",
        },
    ]

    for i, input_data in enumerate(sample_inputs, 1):
        print(f"\n--- Processing Podcast {i}: {input_data['intro']} ---")
        try:
            result = flow.run(
                summary=input_data["summary"],
                intro=input_data["intro"],
                outro=input_data["outro"],
            )
            print(
                "Intro:",
                result.get("intro", "N/A")[:100] + "..."
                if result.get("intro")
                else "N/A",
            )
            print(
                "Main:",
                result.get("main", "N/A")[:100] + "..."
                if result.get("main")
                else "N/A",
            )
            print(
                "Outro:",
                result.get("outro", "N/A")[:100] + "..."
                if result.get("outro")
                else "N/A",
            )
        except Exception as e:
            print(f"✗ Flow execution failed: {e}")
            print("This is expected without proper API keys")

    print("\n=== Demo Complete ===")
    print("\nKey takeaways from this example:")
    print("• Podcast flow configuration with dataclass")
    print("• YAML-based prompt templates for intro, main, and outro")
    print("• Multiple LLM blocks for different content types")
    print("• Python-based orchestration logic")
    print("• Easy to customize and extend for different podcast styles")


if __name__ == "__main__":
    main()
