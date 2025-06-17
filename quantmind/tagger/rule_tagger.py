"""Rule-based tagger for content classification."""

import re
from typing import Dict, List, Optional, Any

from quantmind.models.paper import Paper
from quantmind.tagger.base import BaseTagger
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class RuleTagger(BaseTagger):
    """Rule-based tagger using keyword matching and pattern recognition.

    Uses predefined rules and keyword patterns to classify papers into
    categories and extract relevant tags.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize rule-based tagger.

        Args:
            config: Configuration dictionary with:
                - category_rules: Dict mapping categories to keyword lists
                - tag_patterns: Dict mapping tag types to regex patterns
                - case_sensitive: Whether matching is case sensitive (default: False)
        """
        super().__init__(config)
        self.case_sensitive = self.config.get("case_sensitive", False)

        # Default financial categories and keywords
        self.category_rules = self.config.get(
            "category_rules",
            {
                "Machine Learning in Finance": [
                    "machine learning",
                    "ml",
                    "neural network",
                    "deep learning",
                    "artificial intelligence",
                    "predictive model",
                    "feature engineering",
                    "ensemble method",
                    "random forest",
                    "gradient boosting",
                    "svm",
                ],
                "Deep Learning in Finance": [
                    "deep learning",
                    "neural network",
                    "cnn",
                    "rnn",
                    "lstm",
                    "gru",
                    "transformer",
                    "attention mechanism",
                    "autoencoder",
                    "gan",
                    "deep neural network",
                    "convolutional",
                    "recurrent",
                ],
                "Reinforcement Learning in Finance": [
                    "reinforcement learning",
                    "rl",
                    "q-learning",
                    "policy gradient",
                    "actor-critic",
                    "markov decision",
                    "mdp",
                    "multi-agent",
                    "trading agent",
                    "algorithmic trading",
                    "automated trading",
                ],
                "Time Series Forecasting": [
                    "time series",
                    "forecasting",
                    "prediction",
                    "arima",
                    "garch",
                    "volatility modeling",
                    "trend analysis",
                    "seasonality",
                    "econometric",
                    "financial time series",
                    "stock price prediction",
                ],
                "Risk Management": [
                    "risk management",
                    "var",
                    "value at risk",
                    "stress testing",
                    "credit risk",
                    "market risk",
                    "operational risk",
                    "portfolio risk",
                    "risk assessment",
                    "risk modeling",
                    "basel",
                ],
                "Portfolio Optimization": [
                    "portfolio optimization",
                    "asset allocation",
                    "mean variance",
                    "markowitz",
                    "sharpe ratio",
                    "efficient frontier",
                    "risk parity",
                    "portfolio construction",
                    "rebalancing",
                    "diversification",
                ],
                "Quantitative Trading": [
                    "algorithmic trading",
                    "quantitative trading",
                    "high frequency",
                    "market making",
                    "execution algorithm",
                    "order flow",
                    "market microstructure",
                    "trading strategy",
                    "alpha generation",
                ],
            },
        )

        # Default tag patterns
        self.tag_patterns = self.config.get(
            "tag_patterns",
            {
                "algorithm": r"\b(?:algorithm|model|method|approach|technique)\b",
                "finance": r"\b(?:financial|finance|market|trading|investment|portfolio)\b",
                "quantitative": r"\b(?:quantitative|quant|mathematical|statistical|numerical)\b",
                "prediction": r"\b(?:predict|forecast|estimate|model|analyze)\b",
            },
        )

    def tag_paper(self, paper: Paper) -> Paper:
        """Add tags and categories to a paper using rule-based matching.

        Args:
            paper: Paper object to tag

        Returns:
            Paper object with added tags and categories
        """
        # Combine title and abstract for analysis
        text = f"{paper.title} {paper.abstract}"

        # Extract categories
        categories = self.extract_categories(text, paper.title)
        for category in categories:
            paper.add_category(category)

        # Extract tags
        tags = self.extract_tags(text, paper.title)
        for tag in tags:
            paper.add_tag(tag)

        # Add processing metadata
        paper.meta_info["tagger"] = self.name
        paper.meta_info["tagging_method"] = "rule_based"

        logger.debug(
            f"Tagged paper {paper.get_primary_id()} with {len(categories)} categories and {len(tags)} tags"
        )

        return paper

    def extract_categories(self, text: str, title: str = "") -> List[str]:
        """Extract categories using keyword matching.

        Args:
            text: Text content to analyze
            title: Optional title for additional context

        Returns:
            List of category strings
        """
        if not self.case_sensitive:
            text = text.lower()
            title = title.lower()

        full_text = f"{title} {text}"
        matched_categories = []

        for category, keywords in self.category_rules.items():
            for keyword in keywords:
                if not self.case_sensitive:
                    keyword = keyword.lower()

                if keyword in full_text:
                    matched_categories.append(category)
                    break  # One match per category is enough

        return self.validate_categories(matched_categories)

    def extract_tags(self, text: str, title: str = "") -> List[str]:
        """Extract tags using pattern matching.

        Args:
            text: Text content to analyze
            title: Optional title for additional context

        Returns:
            List of tag strings
        """
        full_text = f"{title} {text}"
        matched_tags = []

        # Apply pattern-based tag extraction
        for tag_type, pattern in self.tag_patterns.items():
            flags = re.IGNORECASE if not self.case_sensitive else 0
            if re.search(pattern, full_text, flags):
                matched_tags.append(tag_type)

        # Extract specific financial terms
        financial_terms = self._extract_financial_terms(full_text)
        matched_tags.extend(financial_terms)

        return self.validate_tags(matched_tags)

    def validate_categories(self, categories: List[str]) -> List[str]:
        """Validate and clean extracted categories, preserving original case.

        Args:
            categories: List of raw categories

        Returns:
            List of validated and cleaned categories
        """
        valid_categories = []
        for category in categories:
            if isinstance(category, str) and len(category.strip()) > 0:
                cleaned_category = (
                    category.strip()
                )  # Keep original case for categories
                if cleaned_category not in valid_categories:
                    valid_categories.append(cleaned_category)
        return valid_categories

    def _extract_financial_terms(self, text: str) -> List[str]:
        """Extract specific financial and quantitative terms.

        Args:
            text: Text to analyze

        Returns:
            List of extracted terms
        """
        terms = []

        # Common financial instruments
        instruments = [
            "stock",
            "bond",
            "option",
            "future",
            "derivative",
            "etf",
            "mutual fund",
        ]
        for instrument in instruments:
            pattern = rf"\b{instrument}s?\b"
            flags = re.IGNORECASE if not self.case_sensitive else 0
            if re.search(pattern, text, flags):
                terms.append(instrument)

        # Market types
        markets = [
            "equity",
            "fixed income",
            "forex",
            "commodity",
            "cryptocurrency",
        ]
        for market in markets:
            pattern = rf"\b{market}\b"
            flags = re.IGNORECASE if not self.case_sensitive else 0
            if re.search(pattern, text, flags):
                terms.append(market.replace(" ", "_"))

        # Statistical methods
        methods = [
            "regression",
            "correlation",
            "cointegration",
            "volatility",
            "momentum",
        ]
        for method in methods:
            pattern = rf"\b{method}\b"
            flags = re.IGNORECASE if not self.case_sensitive else 0
            if re.search(pattern, text, flags):
                terms.append(method)

        return terms

    def get_confidence_score(self, paper: Paper) -> float:
        """Get confidence score based on number of matches.

        Args:
            paper: Tagged paper object

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Simple heuristic: more tags/categories = higher confidence
        total_matches = len(paper.categories) + len(paper.tags)

        if total_matches == 0:
            return 0.0
        elif total_matches <= 2:
            return 0.3
        elif total_matches <= 5:
            return 0.6
        else:
            return 0.8

    def add_category_rule(self, category: str, keywords: List[str]) -> None:
        """Add a new category rule.

        Args:
            category: Category name
            keywords: List of keywords that indicate this category
        """
        self.category_rules[category] = keywords
        logger.info(
            f"Added category rule for '{category}' with {len(keywords)} keywords"
        )

    def add_tag_pattern(self, tag_type: str, pattern: str) -> None:
        """Add a new tag pattern.

        Args:
            tag_type: Type of tag to extract
            pattern: Regex pattern to match
        """
        self.tag_patterns[tag_type] = pattern
        logger.info(f"Added tag pattern for '{tag_type}': {pattern}")

    def get_category_stats(self, papers: List[Paper]) -> Dict[str, int]:
        """Get statistics on category assignments.

        Args:
            papers: List of papers to analyze

        Returns:
            Dictionary mapping categories to counts
        """
        category_counts = {}
        for paper in papers:
            for category in paper.categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
