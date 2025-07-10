"""
Simple example of QuantMind Template system usage.
"""

from quantmind.utils import T


def example_basic_usage():
    """Simple template usage example."""
    
    # Load and render a prompt template
    prompt = T("examples.utils.prompts:paper_analysis").r(
        title="Deep Learning for Financial Time Series Forecasting",
        abstract="This paper proposes a novel LSTM-based approach for predicting stock prices using high-frequency data.",
        keywords="deep learning, LSTM, financial forecasting, high-frequency trading"
    )
    
    print("Generated Prompt:")
    print(prompt)


def example_relative_path():
    """Test relative path loading."""
    
    # Test loading from local.yaml in the same directory
    local_prompt = T(".prompts:local_prompt").r(
        custom_param="test value",
        local_context="relative path test"
    )
    
    print("\nRelative Path Test:")
    print(local_prompt)
    
    # Test nested key access
    nested_prompt = T(".prompts:nested_test.key1").r(var1="nested value")
    
    print("\nNested Key Test:")
    print(nested_prompt)
    
    # Test conditional template
    conditional_prompt = T(".prompts:conditional_test").r(
        condition=True,
        value="conditional test"
    )
    
    print("\nConditional Test:")
    print(conditional_prompt)


if __name__ == "__main__":
    example_basic_usage()
    example_relative_path() 