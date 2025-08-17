"""Podcast Flow - Generate final podcast scripts from summary input.

This flow takes a summary and generates a podcast script in JSON format.
"""

from typing import Any, Dict, List
import json

from quantmind.flow.base import BaseFlow
from quantmind.config.flows import PodcastFlowConfig
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class PodcastFlow(BaseFlow):
    """Flow for generating podcast scripts from summary input."""

    def __init__(self, config: PodcastFlowConfig):
        super().__init__(config)
        self.config = config

    def run(self, summary: str) -> Dict[str, Any]:
        """Execute the podcast script generation flow.

        Args:
            summary: Summary of the podcast content to generate the script from.

        Returns:
            JSON string containing the podcast script
        """
        if summary:
            self.config.summary_hint = summary
            logger.info(f"Using input summary.")
        else:
            logger.warning("No summary provided, using default summary hint.")

        logger.info("Starting podcast script generation flow")
        # Generate podcast script
        script = self._generate_script(self.config.summary_hint)

        logger.info("Podcast script generation completed")
        return script

    def _generate_script(self, summary: str) -> Dict[str, Any]:
        """Generate podcast script from summary."""
        script = {}

        main_generator = self._llm_blocks["main_generator"]
        main_prompt = self._render_prompt("main_prompt", summary_hint=summary)
        main_script = main_generator.generate_text(main_prompt)
        script["main"] = main_script

        return script
