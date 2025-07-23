"""Simple LLM-based tagger for financial research papers using LLMBlock."""

import json
from typing import List

from quantmind.config import LLMTaggerConfig
from quantmind.llm import create_llm_block
from quantmind.models import Paper
from quantmind.utils.logger import get_logger

from .base import BaseTagger

logger = get_logger(__name__)


class LLMTagger(BaseTagger):
    """Simple LLM-based tagger for financial research papers.

    Uses LLMBlock to generate relevant tags for quantitative finance papers.
    """

    def __init__(
        self,
        config: LLMTaggerConfig = None,
    ):
        """Initialize LLM tagger.

        Args:
            config: Configuration for the LLM tagger
        """
        super().__init__()
        self.config = config or LLMTaggerConfig()

        # Create LLMBlock directly from the embedded LLMConfig
        try:
            self.llm_block = create_llm_block(self.config.llm_config)
            logger.info(
                f"Initialized LLM tagger with model: {self.config.llm_config.model}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM block: {e}")
            self.llm_block = None

    def tag_paper(self, paper: Paper) -> Paper:
        """Generate tags for a paper using LLM analysis.

        Args:
            paper: Paper object to tag

        Returns:
            Paper object with added tags
        """
        if not self.llm_block:
            logger.warning("No LLM block available, skipping tagging")
            return paper

        try:
            # Get paper content for analysis
            content = self._prepare_content(paper)

            # Generate tags using LLM
            tags = self._generate_tags(content)

            # Add tags to paper
            for tag in tags:
                paper.add_tag(tag)

            # Store tagging metadata
            paper.meta_info.update(
                {
                    "tagger": "llm_tagger",
                    "model_used": self.config.llm_config.model,
                    "tags_generated": len(tags),
                }
            )

            logger.info(f"Generated {len(tags)} tags for paper: {paper.title}")

        except Exception as e:
            logger.error(f"Error tagging paper {paper.get_primary_id()}: {e}")

        return paper

    def _prepare_content(self, paper: Paper) -> str:
        """Prepare paper content for LLM analysis.

        Args:
            paper: Paper object

        Returns:
            Formatted content string
        """
        content_parts = []

        if paper.title:
            content_parts.append(f"Title: {paper.title}")

        if paper.abstract:
            content_parts.append(f"Abstract: {paper.abstract}")

        # Use first max_tokens characters of full text to stay within token limits
        if paper.full_text:
            content_parts.append(
                f"Content: {paper.full_text[: self.config.llm_config.max_tokens]}..."
            )

        return "\n\n".join(content_parts)

    def _generate_tags(self, content: str) -> List[str]:
        """Generate tags using LLM.

        Args:
            content: Paper content to analyze

        Returns:
            List of generated tags
        """
        prompt = self._build_prompt(content)

        try:
            response = self.llm_block.generate_text(prompt)

            if not response:
                logger.error("No response from LLM")
                return []

            logger.debug(f"LLM response: {response}")

            # Parse tags from response
            tags = self._parse_tags(response)

            # Limit to max_tags
            return tags[: self.config.max_tags]

        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            return []

    def _build_prompt(self, content: str) -> str:
        """Build prompt for tag generation.

        Args:
            content: Paper content

        Returns:
            Formatted prompt
        """
        if self.config.custom_prompt:
            return self.config.custom_prompt.format(
                content=content, max_tags=self.config.max_tags
            )

        # Default prompt for quantitative finance papers
        return f"""Analyze this quantitative finance research paper and generate {self.config.max_tags} relevant tags.

Paper Content:
{content}

Generate tags that capture the key aspects like:
- Market types (equity, forex, crypto, bonds)
- Methods (machine learning, deep learning, statistical)
- Applications (trading, risk management, portfolio optimization)
- Data types (price data, news, sentiment)
- Techniques (LSTM, transformers, regression)

Return only a JSON list of tags, no other text:
["tag1", "tag2", "tag3", "tag4", "tag5"]"""

    def _parse_tags(self, response: str) -> List[str]:
        """Parse tags from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            List of parsed tags
        """
        try:
            # Try to find JSON array in response
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                tags = json.loads(json_str)

                if isinstance(tags, list):
                    # Clean and validate tags
                    cleaned_tags = []
                    for tag in tags:
                        if isinstance(tag, str) and tag.strip():
                            cleaned_tags.append(tag.strip().lower())

                    return cleaned_tags

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON tags: {e}")

        # Fallback: try to extract tags from plain text
        return self._extract_tags_from_text(response)

    def _extract_tags_from_text(self, text: str) -> List[str]:
        """Extract tags from plain text response as fallback.

        Args:
            text: Response text

        Returns:
            List of extracted tags
        """
        # Simple extraction: look for quoted words or comma-separated items
        import re

        # Try to find quoted items first
        quoted_items = re.findall(r'"([^"]*)"', text)
        if quoted_items:
            return [
                item.strip().lower() for item in quoted_items if item.strip()
            ]

        # Try comma-separated items
        lines = text.split("\n")
        for line in lines:
            if "," in line and not line.startswith("#"):
                items = [item.strip().lower() for item in line.split(",")]
                if len(items) >= 2:
                    return [item for item in items if item and len(item) > 1]

        logger.warning("Could not extract tags from response")
        return []

    def extract_tags(self, text: str, title: str = "") -> List[str]:
        """Extract tags from arbitrary text.

        Args:
            text: Text content to analyze
            title: Optional title for context

        Returns:
            List of extracted tags
        """
        if not self.llm_block:
            return []

        content = f"Title: {title}\n\nContent: {text}" if title else text
        return self._generate_tags(content)

    def test_connection(self) -> bool:
        """Test if the LLM connection is working.

        Returns:
            True if connection is working, False otherwise
        """
        if not self.llm_block:
            return False

        return self.llm_block.test_connection()

    @property
    def llm_type(self) -> str:
        """Get the LLM type."""
        return "openai"  # Default, can be made configurable if needed

    @property
    def llm_name(self) -> str:
        """Get the LLM name."""
        return self.config.llm_config.model
