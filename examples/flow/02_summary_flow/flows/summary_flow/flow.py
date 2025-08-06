"""Summary flow with mock LLM for demonstration."""

from typing import List
from quantmind.config.flows import SummaryFlowConfig
from quantmind.flow.summary_flow import SummaryFlow


class MockLLMBlock:
    """Mock LLM block for demonstration purposes."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        """Generate mock responses based on model type."""
        if (
            "cheap_summarizer" in self.model_name
            or "gpt-4o-mini" in self.model_name
        ):
            # Mock cheap model response - shorter, more focused
            if "methodology" in prompt.lower():
                return "This section discusses ML algorithms for financial prediction using historical data."
            elif "results" in prompt.lower():
                return (
                    "The model achieved 67% accuracy with Sharpe ratio of 1.8."
                )
            else:
                return "Key findings: Machine learning shows promise for financial applications."

        elif (
            "powerful_combiner" in self.model_name
            or "gpt-4o" in self.model_name
        ):
            # Mock powerful model response - comprehensive combination
            if "combine" in prompt.lower() or "summaries" in prompt.lower():
                return (
                    "## Comprehensive Summary\n\n"
                    "This research demonstrates the successful application of machine learning "
                    "techniques in quantitative finance. The study employs various ML algorithms "
                    "including random forests and neural networks to predict market movements.\n\n"
                    "**Key Achievements:**\n"
                    "- Achieved 67% directional prediction accuracy\n"
                    "- Sharpe ratio improvement to 1.8 vs 1.2 baseline\n"
                    "- Demonstrated superiority over traditional statistical models\n\n"
                    "**Methodology:** The approach combines technical indicators with fundamental "
                    "analysis metrics, processed through ensemble learning methods.\n\n"
                    "**Implications:** These results suggest significant potential for ML-driven "
                    "trading strategies in institutional finance applications."
                )
            else:
                return (
                    "This comprehensive analysis of machine learning applications in finance "
                    "demonstrates significant improvements over traditional approaches, with "
                    "practical implications for algorithmic trading and risk management."
                )

        return "Mock response generated successfully."


class DemoSummaryFlow(SummaryFlow):
    """Summary flow with mock LLM blocks for demonstration."""

    def _initialize_llm_blocks(self, llm_configs):
        """Override to use mock LLM blocks."""
        llm_blocks = {}
        for identifier, llm_config in llm_configs.items():
            # Create mock LLM blocks instead of real ones
            mock_llm = MockLLMBlock(f"{identifier}_{llm_config.model}")
            llm_blocks[identifier] = mock_llm
            print(
                f"✓ Initialized mock LLM '{identifier}' with model: {llm_config.model}"
            )

        return llm_blocks
