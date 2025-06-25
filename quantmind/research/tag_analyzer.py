"""LLM-based tag analyzer for research papers."""

import json
from typing import Dict, List, Optional, Tuple, Any

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType
    from camel.agents import ChatAgent

    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False

from quantmind.models.paper import Paper
from quantmind.models.analysis import PaperTag, AnalysisConfig
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class LLMTagAnalyzer:
    """LLM-based tag analyzer for hierarchical paper classification.

    Uses large language models to extract structured tags covering:
    - Market types (equity, forex, crypto, etc.)
    - Trading frequency (high-frequency, daily, etc.)
    - Algorithm models (LSTM, Transformer, etc.)
    - Applications and domains
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize LLM tag analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or AnalysisConfig()
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize the LLM client."""
        if not CAMEL_AVAILABLE:
            logger.warning("CAMEL library not available")
            return

        try:
            model_instance = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O,
                model_config_dict={
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            )
            self.client = ChatAgent(model=model_instance)
            logger.info("Initialized CAMEL LLM client for tag analysis")
        except Exception as e:
            logger.error(f"Failed to initialize CAMEL client: {e}")
            self.client = None

    def analyze_paper(
        self, paper: Paper
    ) -> Tuple[List[PaperTag], List[PaperTag]]:
        """Analyze paper to extract primary and secondary tags using LLM.

        Args:
            paper: Paper object to analyze

        Returns:
            Tuple of (primary_tags, secondary_tags)
        """
        logger.info(f"Analyzing tags for paper: {paper.title}")

        if not self.client:
            logger.warning("No LLM client available, returning empty tags")
            return [], []

        try:
            # Get analysis text
            text = self._get_analysis_text(paper)

            # Get LLM classification
            classification = self._get_llm_classification(paper.title, text)

            # Extract tags from classification
            primary_tags, secondary_tags = (
                self._extract_tags_from_classification(classification)
            )

            logger.info(
                f"Extracted {len(primary_tags)} primary and {len(secondary_tags)} secondary tags"
            )

            return primary_tags, secondary_tags

        except Exception as e:
            logger.error(f"Error in LLM tag analysis: {e}")
            return [], []

    def _get_analysis_text(self, paper: Paper) -> str:
        """Get combined text for analysis.

        Args:
            paper: Paper object

        Returns:
            Combined text for analysis
        """
        text_parts = []

        if paper.title:
            text_parts.append(f"Title: {paper.title}")

        if paper.abstract:
            text_parts.append(f"Abstract: {paper.abstract}")

        if paper.full_text:
            # Use first 3000 characters of full text to avoid too much noise
            text_parts.append(f"Content: {paper.full_text[:3000]}...")

        return "\n\n".join(text_parts)

    def _get_llm_classification(self, title: str, text: str) -> Dict[str, Any]:
        """Get classification from LLM.

        Args:
            title: Paper title
            text: Analysis text

        Returns:
            Classification dictionary
        """
        prompt = self._create_tag_prompt(text)

        try:
            response = self.client.step(prompt)
            response_content = response.msgs[0].content.strip()

            logger.debug(f"Raw LLM response: {response_content}")

            # Parse JSON response
            try:
                # Try to extract JSON from response if it's not pure JSON
                json_start = response_content.find("{")
                json_end = response_content.rfind("}") + 1

                if json_start != -1 and json_end > json_start:
                    json_content = response_content[json_start:json_end]
                    classification = json.loads(json_content)
                    return classification
                else:
                    # Try parsing the entire response as JSON
                    classification = json.loads(response_content)
                    return classification

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Response content: {response_content[:500]}...")
                return self._get_default_classification()

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return self._get_default_classification()

    def _create_tag_prompt(self, content: str) -> str:
        """Create prompt for tag extraction."""
        max_content_length = 8000
        if len(content) > max_content_length:
            content = (
                content[:max_content_length]
                + "\n\n[Content truncated for length]"
            )

        prompt = f"""Analyze this research paper content and extract hierarchical tags. Focus on the most important aspects.

Paper Content:
{content}

Extract tags in this JSON format ONLY:
{{
  "primary_tags": [
    {{
      "tag": "market_type",
      "value": "equity/forex/crypto/commodity",
      "confidence": 0.9
    }},
    {{
      "tag": "frequency",
      "value": "high_frequency/intraday/daily/weekly",
      "confidence": 0.8
    }},
    {{
      "tag": "algorithm_type",
      "value": "ml/deep_learning/reinforcement_learning/statistical",
      "confidence": 0.9
    }},
    {{
      "tag": "model_family",
      "value": "transformer/cnn/rnn/lstm/bert/gpt",
      "confidence": 0.8
    }},
    {{
      "tag": "application",
      "value": "prediction/classification/clustering/optimization",
      "confidence": 0.7
    }}
  ],
  "secondary_tags": [
    {{
      "tag": "specific_technique",
      "value": "attention/convolution/recurrent/ensemble",
      "confidence": 0.8
    }},
    {{
      "tag": "data_type",
      "value": "price/volume/orderbook/news/sentiment",
      "confidence": 0.7
    }},
    {{
      "tag": "evaluation_metric",
      "value": "sharpe_ratio/sortino/max_drawdown/accuracy",
      "confidence": 0.6
    }},
    {{
      "tag": "market_regime",
      "value": "trending/mean_reverting/volatile/stable",
      "confidence": 0.6
    }},
    {{
      "tag": "risk_management",
      "value": "position_sizing/stop_loss/portfolio_optimization",
      "confidence": 0.5
    }}
  ]
}}

Rules:
- Return ONLY valid JSON, no other text
- Use confidence scores 0.5-1.0
- Extract up to {self.config.max_primary_tags} primary tags
- Extract up to {self.config.max_secondary_tags} secondary tags
- Focus on the most relevant and specific tags
- If information is unclear, use lower confidence scores

JSON Response:"""

        return prompt

    def _extract_tags_from_classification(
        self, classification: Dict[str, Any]
    ) -> Tuple[List[PaperTag], List[PaperTag]]:
        """Extract tags from LLM classification results.

        Args:
            classification: Classification dictionary from LLM

        Returns:
            Tuple of (primary_tags, secondary_tags)
        """
        primary_tags = []
        secondary_tags = []

        # Extract primary tags
        primary_data = classification.get("primary_tags", [])
        if isinstance(primary_data, list):
            for tag_data in primary_data:
                if isinstance(tag_data, dict):
                    tag = PaperTag(
                        tag=tag_data.get("tag", tag_data.get("name", "")),
                        value=tag_data.get(
                            "value", tag_data.get("category", "")
                        ),
                        confidence=tag_data.get("confidence", 0.8),
                    )
                    if tag.tag and tag.value:
                        primary_tags.append(tag)

        # Extract secondary tags
        secondary_data = classification.get("secondary_tags", [])
        if isinstance(secondary_data, list):
            for tag_data in secondary_data:
                if isinstance(tag_data, dict):
                    tag = PaperTag(
                        tag=tag_data.get("tag", tag_data.get("name", "")),
                        value=tag_data.get(
                            "value", tag_data.get("category", "")
                        ),
                        confidence=tag_data.get("confidence", 0.8),
                    )
                    if tag.tag and tag.value:
                        secondary_tags.append(tag)

        # Limit number of tags
        primary_tags = primary_tags[: self.config.max_primary_tags]
        secondary_tags = secondary_tags[: self.config.max_secondary_tags]

        return primary_tags, secondary_tags

    def _get_default_classification(self) -> Dict[str, Any]:
        """Get default classification when LLM analysis fails.

        Returns:
            Default classification dictionary
        """
        return {
            "primary_tags": [],
            "secondary_tags": [],
            "summary": "Analysis failed - no tags extracted",
        }

    def get_tag_hierarchy(
        self, primary_tags: List[PaperTag], secondary_tags: List[PaperTag]
    ) -> Dict[str, List[PaperTag]]:
        """Organize tags into hierarchical structure.

        Args:
            primary_tags: Level 1 tags
            secondary_tags: Level 2 tags

        Returns:
            Dictionary with hierarchical tag structure
        """
        hierarchy = {
            "markets": [],
            "frequencies": [],
            "algorithms": [],
            "applications": [],
            "data_types": [],
            "strategies": [],
            "methods": [],
        }

        all_tags = primary_tags + secondary_tags

        for tag in all_tags:
            if tag.tag == "market":
                hierarchy["markets"].append(tag)
            elif tag.tag == "frequency":
                hierarchy["frequencies"].append(tag)
            elif tag.tag == "algorithm":
                hierarchy["algorithms"].append(tag)
            elif tag.tag == "application":
                hierarchy["applications"].append(tag)
            elif tag.tag == "data_type":
                hierarchy["data_types"].append(tag)
            elif tag.tag == "strategy":
                hierarchy["strategies"].append(tag)
            elif tag.tag == "method":
                hierarchy["methods"].append(tag)

        return hierarchy

    def generate_tag_summary(
        self, primary_tags: List[PaperTag], secondary_tags: List[PaperTag]
    ) -> str:
        """Generate human-readable tag summary.

        Args:
            primary_tags: Level 1 tags
            secondary_tags: Level 2 tags

        Returns:
            Formatted tag summary
        """
        if not primary_tags and not secondary_tags:
            return "No specific tags identified."

        summary_parts = []

        if primary_tags:
            primary_names = [tag.tag for tag in primary_tags]
            summary_parts.append(f"Primary focus: {', '.join(primary_names)}")

        if secondary_tags:
            secondary_names = [tag.tag for tag in secondary_tags]
            summary_parts.append(
                f"Secondary aspects: {', '.join(secondary_names)}"
            )

        return " | ".join(summary_parts)

    def _parse_tag_response(
        self, response_data: Dict[str, Any]
    ) -> Dict[str, List[PaperTag]]:
        """Parse tag response from LLM.

        Args:
            response_data: Tag data from LLM

        Returns:
            Dictionary with primary_tags and secondary_tags lists
        """
        primary_tags = []
        secondary_tags = []

        # Parse primary tags
        primary_data = response_data.get("primary_tags", [])
        if isinstance(primary_data, list):
            for tag_data in primary_data:
                if isinstance(tag_data, dict):
                    # Map LLM response fields to model fields
                    tag = PaperTag(
                        tag=tag_data.get("tag", tag_data.get("name", "")),
                        value=tag_data.get(
                            "value", tag_data.get("category", "")
                        ),
                        confidence=tag_data.get("confidence", 0.8),
                    )
                    if tag.tag and tag.value:
                        primary_tags.append(tag)

        # Parse secondary tags
        secondary_data = response_data.get("secondary_tags", [])
        if isinstance(secondary_data, list):
            for tag_data in secondary_data:
                if isinstance(tag_data, dict):
                    # Map LLM response fields to model fields
                    tag = PaperTag(
                        tag=tag_data.get("tag", tag_data.get("name", "")),
                        value=tag_data.get(
                            "value", tag_data.get("category", "")
                        ),
                        confidence=tag_data.get("confidence", 0.8),
                    )
                    if tag.tag and tag.value:
                        secondary_tags.append(tag)

        return {"primary_tags": primary_tags, "secondary_tags": secondary_tags}
