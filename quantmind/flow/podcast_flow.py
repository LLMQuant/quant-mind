"""Podcast Flow - Generate final podcast scripts from summary input.

This flow takes a summary and generates a podcast script in JSON format.
"""

from typing import Any, Dict, List
import json

from quantmind.flow.base import BaseFlow
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class PodcastFlow(BaseFlow):
    """Flow for generating podcast scripts from summary input."""

    def run(self, summary: str, intro: str = "", outro: str = "") -> dict:
        """Execute the podcast script generation flow.

        Args:
            summary: Summary text or content to convert to podcast script
            intro: Intro hint for the podcast script
            outro: Outro hint for the podcast script
        Returns:
            JSON string containing the podcast script
        """
        logger.info("Starting podcast script generation flow")

        # Generate podcast script
        script_str = self._generate_script(summary, intro, outro)
        script = json.loads(script_str)

        logger.info("Podcast script generation completed")
        return script

    def _generate_script(
        self, summary: str, intro: str = "", outro: str = ""
    ) -> Dict[str, Any]:
        """Generate podcast script from summary."""
        script = {}

        if self._llm_blocks.get(
            "intro_generator", None
        ) and self.config.prompt_templates.get("intro_prompt", None):
            intro_generator = self._llm_blocks["intro_generator"]
            intro_prompt = self._render_prompt("intro_prompt", intro_hint=intro)
            intro_script = intro_generator.generate_text(intro_prompt)
            script["intro"] = intro_script

        if self._llm_blocks.get(
            "outro_generator", None
        ) and self.config.prompt_templates.get("outro_prompt", None):
            outro_generator = self._llm_blocks["outro_generator"]
            outro_prompt = self._render_prompt("outro_prompt", outro_hint=outro)
            outro_script = outro_generator.generate_text(outro_prompt)
            script["outro"] = outro_script

        main_generator = self._llm_blocks["main_generator"]
        main_prompt = self._render_prompt("main_prompt", summary=summary)
        main_script = main_generator.generate_text(main_prompt)
        script["main"] = main_script

        return script
