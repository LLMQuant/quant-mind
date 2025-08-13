"""Podcast flow with mock LLM for demonstration."""

from typing import Dict
from quantmind.config.flows import BaseFlowConfig
from quantmind.flow.podcast_flow import PodcastFlow


class MockLLMBlock:
    """Mock LLM block for demonstration purposes."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        """Generate mock responses based on model type and prompt content."""
        if "intro_generator" in self.model_name:
            # Mock intro generator response
            if "AI" in prompt or "artificial intelligence" in prompt.lower():
                return "Welcome to our podcast! Today we're talking about AI."
            elif "quantum" in prompt.lower():
                return "Welcome to our podcast! Today we're talking about quantum computing."
            elif "climate" in prompt.lower():
                return "Welcome to our podcast! Today we're talking about climate change."
            else:
                return (
                    "Welcome to our podcast! Today we're talking about a topic."
                )

        elif "main_generator" in self.model_name:
            # Mock main content generator response
            if "AI" in prompt or "artificial intelligence" in prompt.lower():
                return "Artificial Intelligence is great."
            elif "quantum" in prompt.lower():
                return "Quantum computing is great."
            elif "climate" in prompt.lower():
                return "Climate change is great."
            else:
                return "This is great content."

        elif "outro_generator" in self.model_name:
            # Mock outro generator response
            if "AI" in prompt or "artificial intelligence" in prompt.lower():
                return "That wraps up our deep dive into AI."
            elif "quantum" in prompt.lower():
                return "That concludes our exploration of quantum computing."
            elif "climate" in prompt.lower():
                return "That brings us to the end of our climate change discussion."
            else:
                return "That wraps up another episode."

        return "Mock response generated successfully."


class DemoPodcastFlow(PodcastFlow):
    """Podcast flow with mock LLM blocks for demonstration."""

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
