"""Podcast flow with mock LLM for demonstration."""

from typing import Dict
from quantmind.config.flows import BaseFlowConfig
from quantmind.flow.podcast_flow import PodcastFlow
from typing import Literal

class MockLLMBlock:
    """Mock LLM block for demonstration purposes."""

    def __init__(self, model_name: str, role: Literal["intro", "main", "outro"]):
        self.model_name = model_name
        self.role = role

    def generate_text(self, prompt: str) -> str:
        """Generate mock responses based on model type and prompt content."""
        p = prompt or ""
        pl = p.lower()

        if self.role == "intro":
            # Mock intro generator response
            if "ai" in pl or "artificial intelligence" in pl:
                return "Welcome to our podcast! Today we're talking about AI."
            elif "quantum" in pl:
                return "Welcome to our podcast! Today we're talking about quantum computing."
            elif "climate" in pl:
                return "Welcome to our podcast! Today we're talking about climate change."
            else:
                return (
                    "Welcome to our podcast! Today we're talking about a topic."
                )

        elif self.role == "main":
            # Mock main content generator response
            if "ai" in pl or "artificial intelligence" in pl:
                return "Artificial Intelligence is great."
            elif "quantum" in pl:
                return "Quantum computing is great."
            elif "climate" in pl:
                return "Climate change is great."
            else:
                return "This is great content."

        elif self.role == "outro":
            # Mock outro generator response
            if "ai" in pl or "artificial intelligence" in pl:
                return "That wraps up our deep dive into AI."
            elif "quantum" in pl:
                return "That concludes our exploration of quantum computing."
            elif "climate" in pl:
                return "That brings us to the end of our climate change discussion."
            else:
                return "That wraps up another episode."

        return "Mock response generated successfully."


class DemoPodcastFlow(PodcastFlow):
    """Podcast flow with mock LLM blocks for demonstration."""

    def _initialize_llm_blocks(self, llm_configs):
        """Override to use mock LLM blocks."""
        role_map = {
            "intro_generator": "intro",
            "main_generator": "main",
            "outro_generator": "outro",
        }
        llm_blocks = {}
        for identifier, llm_config in llm_configs.items():
            if identifier not in role_map:
                raise ValueError(f"Unknown LLM block id: {identifier}")
            role = role_map[identifier]
            llm_blocks[identifier] = MockLLMBlock(llm_config.model, role)
            print(
                f"✓ Initialized mock LLM '{identifier}' with model: {llm_config.model}"
            )

        return llm_blocks
