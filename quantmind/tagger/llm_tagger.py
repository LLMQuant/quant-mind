"""LLM-based tagger for advanced content classification."""

import json
from typing import Dict, List, Optional, Any

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType
    from camel.agents import ChatAgent

    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from quantmind.models.paper import Paper
from quantmind.tagger.base import BaseTagger
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class LLMTagger(BaseTagger):
    """LLM-based tagger for advanced content classification.

    Uses large language models to perform sophisticated classification
    and tag extraction with domain-specific prompts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM tagger.

        Args:
            config: Configuration dictionary with:
                - model_type: 'camel' or 'openai' (default: 'openai')
                - model_name: Specific model to use (default: 'gpt-4')
                - temperature: Model temperature (default: 0.0)
                - api_key: API key for OpenAI (if using openai client)
                - max_tokens: Maximum tokens for response (default: 1000)
        """
        super().__init__(config)

        self.model_type = self.config.get("model_type", "openai")
        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.0)
        self.max_tokens = self.config.get("max_tokens", 1000)

        # Initialize the appropriate client
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize the LLM client based on configuration."""
        if self.model_type == "camel" and CAMEL_AVAILABLE:
            try:
                model_instance = ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI,
                    model_type=ModelType.GPT_4O,
                    model_config_dict={"temperature": self.temperature},
                )
                self.client = ChatAgent(model=model_instance)
                logger.info("Initialized CAMEL LLM client")
            except Exception as e:
                logger.error(f"Failed to initialize CAMEL client: {e}")
                self.client = None

        elif self.model_type == "openai" and OPENAI_AVAILABLE:
            try:
                api_key = self.config.get("api_key")
                self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
                logger.info("Initialized OpenAI client")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.warning(
                f"LLM client not available for type: {self.model_type}"
            )
            self.client = None

    def tag_paper(self, paper: Paper) -> Paper:
        """Add tags and categories to a paper using LLM analysis.

        Args:
            paper: Paper object to tag

        Returns:
            Paper object with added tags and categories
        """
        if not self.client:
            logger.warning("No LLM client available, skipping tagging")
            return paper

        try:
            # Get classification from LLM
            classification = self._classify_paper(
                paper.title, paper.abstract, paper.full_text or ""
            )

            # Extract categories from classification
            categories = self._extract_categories_from_classification(
                classification
            )
            for category in categories:
                paper.add_category(category)

            # Extract tags from classification
            tags = self._extract_tags_from_classification(classification)
            for tag in tags:
                paper.add_tag(tag)

            # Store detailed classification in metadata
            paper.meta_info.update(
                {
                    "llm_classification": classification,
                    "tagger": self.name,
                    "tagging_method": "llm_based",
                    "model_used": self.model_name,
                }
            )

            logger.debug(
                f"LLM tagged paper {paper.get_primary_id()} with {len(categories)} categories and {len(tags)} tags"
            )

        except Exception as e:
            logger.error(
                f"Error in LLM tagging for paper {paper.get_primary_id()}: {e}"
            )

        return paper

    def extract_categories(self, text: str, title: str = "") -> List[str]:
        """Extract categories using LLM analysis.

        Args:
            text: Text content to analyze
            title: Optional title for additional context

        Returns:
            List of category strings
        """
        if not self.client:
            return []

        try:
            classification = self._classify_paper(title, text)
            return self._extract_categories_from_classification(classification)
        except Exception as e:
            logger.error(f"Error extracting categories: {e}")
            return []

    def extract_tags(self, text: str, title: str = "") -> List[str]:
        """Extract tags using LLM analysis.

        Args:
            text: Text content to analyze
            title: Optional title for additional context

        Returns:
            List of tag strings
        """
        if not self.client:
            return []

        try:
            classification = self._classify_paper(title, text)
            return self._extract_tags_from_classification(classification)
        except Exception as e:
            logger.error(f"Error extracting tags: {e}")
            return []

    def _classify_paper(
        self, title: str, abstract: str, full_text: str = ""
    ) -> Dict[str, Any]:
        """Classify a paper using LLM.

        Args:
            title: Paper title
            abstract: Paper abstract
            full_text: Optional full text content

        Returns:
            Dictionary containing classification information
        """
        prompt = self._build_classification_prompt(title, abstract, full_text)

        try:
            if self.model_type == "camel":
                response = self.client.step(prompt)
                response_content = response.msgs[0].content.strip()
            else:  # openai
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                response_content = response.choices[0].message.content.strip()

            logger.debug(f"Raw LLM response: {response_content}")

            # Parse JSON response
            try:
                classification = json.loads(response_content)
                return classification
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                return self._get_default_classification()

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return self._get_default_classification()

    def _build_classification_prompt(
        self, title: str, abstract: str, full_text: str = ""
    ) -> str:
        """Build the classification prompt for the LLM.

        Args:
            title: Paper title
            abstract: Paper abstract
            full_text: Optional full text content

        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze this academic paper and provide a detailed classification in JSON format:

Title: {title}
Abstract: {abstract}
{f"Content: {full_text[:2000]}..." if full_text else ""}

Please classify the paper based on the following aspects:

1. **Primary Categories**: Choose the most relevant categories from:
   - Machine Learning in Finance
   - Deep Learning in Finance
   - Reinforcement Learning in Finance
   - Time Series Forecasting
   - Risk Management
   - Portfolio Optimization
   - Quantitative Trading
   - Algorithmic Trading
   - Financial Markets
   - Behavioral Finance

2. **Technical Tags**: Identify specific technical aspects:
   - Models/Algorithms used (e.g., LSTM, Transformer, Random Forest)
   - Data types (e.g., price_data, news_data, sentiment_data)
   - Market types (e.g., equity, forex, crypto, commodity)
   - Trading strategies (e.g., trend_following, mean_reversion, momentum)
   - Analysis methods (e.g., technical_analysis, fundamental_analysis)

3. **Application Domain**: Specific finance application area
4. **Methodology**: Research methodology used
5. **Trading Frequency**: high_frequency, medium_frequency, or low_frequency (if applicable)

Return the classification in this exact JSON format:
{{
    "primary_categories": ["string"],
    "technical_tags": ["string"],
    "application_domain": "string",
    "methodology": "string",
    "trading_frequency": "string",
    "models_used": ["string"],
    "data_types": ["string"],
    "market_types": ["string"],
    "trading_strategies": ["string"]
}}

Focus on being precise and only include information that is clearly present in the paper."""

        return prompt

    def _extract_categories_from_classification(
        self, classification: Dict[str, Any]
    ) -> List[str]:
        """Extract categories from LLM classification results.

        Args:
            classification: Classification dictionary from LLM

        Returns:
            List of category strings
        """
        categories = []

        # Add primary categories
        primary_cats = classification.get("primary_categories", [])
        if isinstance(primary_cats, list):
            categories.extend(primary_cats)
        elif isinstance(primary_cats, str):
            categories.append(primary_cats)

        # Add application domain as category if specified
        domain = classification.get("application_domain")
        if domain and isinstance(domain, str) and domain != "unknown":
            categories.append(domain)

        return self.validate_categories(categories)

    def _extract_tags_from_classification(
        self, classification: Dict[str, Any]
    ) -> List[str]:
        """Extract tags from LLM classification results.

        Args:
            classification: Classification dictionary from LLM

        Returns:
            List of tag strings
        """
        tags = []

        # Extract tags from various classification fields
        tag_fields = [
            "technical_tags",
            "models_used",
            "data_types",
            "market_types",
            "trading_strategies",
        ]

        for field in tag_fields:
            field_value = classification.get(field, [])
            if isinstance(field_value, list):
                tags.extend(field_value)
            elif isinstance(field_value, str) and field_value != "unknown":
                tags.append(field_value)

        # Add methodology and trading frequency as tags
        methodology = classification.get("methodology")
        if methodology and methodology != "unknown":
            tags.append(methodology)

        trading_freq = classification.get("trading_frequency")
        if trading_freq and trading_freq != "unknown":
            tags.append(trading_freq)

        return self.validate_tags(tags)

    def _get_default_classification(self) -> Dict[str, Any]:
        """Get default classification when LLM analysis fails.

        Returns:
            Default classification dictionary
        """
        return {
            "primary_categories": [],
            "technical_tags": [],
            "application_domain": "unknown",
            "methodology": "unknown",
            "trading_frequency": "unknown",
            "models_used": [],
            "data_types": [],
            "market_types": [],
            "trading_strategies": [],
        }

    def get_confidence_score(self, paper: Paper) -> float:
        """Get confidence score for LLM tagging results.

        Args:
            paper: Tagged paper object

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Check if we have LLM classification data
        classification = paper.meta_info.get("llm_classification", {})

        if not classification:
            return 0.0

        # Count non-empty/non-unknown fields
        total_fields = 0
        filled_fields = 0

        for key, value in classification.items():
            total_fields += 1
            if isinstance(value, list) and len(value) > 0:
                filled_fields += 1
            elif isinstance(value, str) and value not in ["unknown", ""]:
                filled_fields += 1

        if total_fields == 0:
            return 0.0

        # Higher confidence for more filled fields
        base_confidence = filled_fields / total_fields

        # Boost confidence if we have both categories and tags
        if len(paper.categories) > 0 and len(paper.tags) > 0:
            base_confidence = min(1.0, base_confidence + 0.2)

        return base_confidence
