"""Example: Using the new unified configuration system.

This example demonstrates how to:
1. Load configuration from YAML with environment variable substitution
2. Create default configuration
3. Save configuration to YAML
4. Access configuration values
"""

import os
from pathlib import Path

from quantmind.config import Setting, create_default_config, load_config


def main():
    """Demonstrate configuration usage."""
    print("QuantMind Configuration System Example")
    print("=" * 40)

    # Example 1: Create default configuration
    print("\n1. Creating default configuration:")
    default_setting = create_default_config()
    print(f"   Source type: {type(default_setting.source).__name__}")
    print(f"   Parser type: {type(default_setting.parser).__name__}")
    print(f"   Storage directory: {default_setting.storage.storage_dir}")

    # Example 2: Save default configuration to YAML
    print("\n2. Saving default configuration to YAML:")
    config_dir = Path("examples/config")
    config_dir.mkdir(exist_ok=True)
    default_config_path = config_dir / "default_config.yaml"
    default_setting.save_to_yaml(default_config_path)
    print(f"   Saved to: {default_config_path}")

    # Example 3: Load configuration from YAML
    print("\n3. Loading configuration from YAML:")
    sample_config_path = config_dir / "sample_config.yaml"

    if sample_config_path.exists():
        try:
            # Load configuration with environment variable substitution
            setting = load_config(sample_config_path)
            print(f"   ✅ Loaded configuration from {sample_config_path}")
            print(f"   Source: {setting.source}")
            print(f"   Parser: {setting.parser}")
            print(f"   Log level: {setting.log_level}")

            # Access specific configuration values
            if setting.source:
                print(f"   Source max_results: {setting.source.max_results}")

            if setting.parser:
                print(f"   Parser method: {setting.parser.method}")

        except Exception as e:
            print(f"   ❌ Failed to load configuration: {e}")
    else:
        print(f"   ⚠️  Sample config not found at {sample_config_path}")

    # Example 4: Environment variable substitution
    print("\n4. Environment variable substitution:")
    print("   Set these environment variables to see substitution in action:")
    print("   - ARXIV_MAX_RESULTS=50")
    print("   - OPENAI_MODEL=gpt-3.5-turbo")
    print("   - LOG_LEVEL=DEBUG")

    env_vars = ["ARXIV_MAX_RESULTS", "OPENAI_MODEL", "LOG_LEVEL"]
    for var in env_vars:
        value = os.getenv(var)
        status = "✅ Set" if value else "❌ Not set"
        print(f"   {var}: {status}" + (f" = {value}" if value else ""))

    # Example 5: Direct configuration creation
    print("\n5. Creating configuration programmatically:")
    from quantmind.config import ArxivSourceConfig, LLMConfig, PDFParserConfig

    custom_setting = Setting(
        source=ArxivSourceConfig(
            max_results=50, sort_by="relevance", download_pdfs=True
        ),
        parser=PDFParserConfig(method="pymupdf", extract_tables=True),
        llm=LLMConfig(model="gpt-4o", temperature=0.3),
        log_level="DEBUG",
    )

    print(f"   ✅ Created custom configuration")
    print(f"   Source max_results: {custom_setting.source.max_results}")
    print(f"   Parser method: {custom_setting.parser.method}")
    print(f"   LLM model: {custom_setting.llm.model}")


if __name__ == "__main__":
    main()
