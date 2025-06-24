"""Simple LLM-based tagger for financial research papers."""

import json
from typing import List

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from quantmind.config import LLMTaggerConfig
from quantmind.models import Paper
from quantmind.utils.logger import get_logger

from .base import BaseTagger

logger = get_logger(__name__)


class LLMTagger(BaseTagger):
    """Simple LLM-based tagger for financial research papers.

    Generates relevant tags for quantitative finance papers using LLM analysis.
    """

    def __init__(
        self,
        config: LLMTaggerConfig = LLMTaggerConfig(),
    ):
        """Initialize LLM tagger.

        Args:
            config: Configuration for the LLM tagger
        """
        super().__init__()

        self.config = config

        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize the LLM client."""
        if self.llm_type == "openai" and OPENAI_AVAILABLE:
            try:
                self.client = (
                    OpenAI(
                        api_key=self.config.api_key,
                        base_url=self.config.base_url,
                    )
                    if self.config.base_url
                    else OpenAI(
                        api_key=self.config.api_key,
                    )
                )
                logger.info(
                    f"Initialized OpenAI client with model {self.llm_name}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.warning(
                f"LLM client not available for type: {self.llm_type}"
            )
            self.client = None

    def tag_paper(self, paper: Paper) -> Paper:
        """Generate tags for a paper using LLM analysis.

        Args:
            paper: Paper object to tag

        Returns:
            Paper object with added tags
        """
        if not self.client:
            logger.warning("No LLM client available, skipping tagging")
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
                    "model_used": self.llm_name,
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
                f"Content: {paper.full_text[: self.config.max_tokens]}..."
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
            response = self.client.chat.completions.create(
                model=self.config.llm_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
            )

            response_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM response: {response_text}")

            # Parse tags from response
            tags = self._parse_tags(response_text)

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

User instructions (if empty, skip this part):
{self.config.custom_instructions}

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
        if not self.client:
            return []

        content = f"Title: {title}\n\nContent: {text}" if title else text
        return self._generate_tags(content)

    @property
    def llm_type(self) -> str:
        """Get the LLM type."""
        return self.config.llm_type

    @property
    def llm_name(self) -> str:
        """Get the LLM name."""
        return self.config.llm_name
