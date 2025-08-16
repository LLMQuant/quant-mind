"""Podcast flow with mock LLM for demonstration."""

from typing import Dict, Any
from venv import create
from quantmind.config.flows import PodcastFlowConfig
from quantmind.flow.podcast_flow import PodcastFlow
from quantmind.llm.block import LLMBlock, create_llm_block


class CustomizedPodcastFlowConfig(PodcastFlowConfig):
    """Configuration for the PodcastFlow with LLM blocks."""

    num_speakers: int = 2
    speaker_languages: str = "en-us"
    summary_hint: str = "This is a sample summary hint for the podcast."


class CustomizedPodcastFlow(PodcastFlow):
    """Podcast flow with mock LLM blocks for demonstration."""

    def _initialize_llm_blocks(self, llm_configs):
        """Override to use mock LLM blocks."""
        llm_blocks = {}
        for identifier, llm_config in llm_configs.items():
            # Create mock LLM blocks instead of real ones
            llm_block = create_llm_block(llm_config)
            llm_blocks[identifier] = llm_block
            print(
                f"✓ Initialized mock LLM '{identifier}' with model: {llm_config.model}"
            )

        return llm_blocks
