"""Paper analysis workflow for tag generation and content analysis."""

import json
from typing import List, Dict, Any, Optional, Tuple

from quantmind.workflow.base import BaseWorkflow
from quantmind.config.workflows import AnalyzerWorkflowConfig
from quantmind.models.paper import Paper

# PaperTag removed - using simple strings
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class AnalyzerWorkflow(BaseWorkflow):
    """Workflow for analyzing research papers and generating structured tags.

    Uses LLM to analyze paper content and generate hierarchical tags covering:
    - Primary categories: market type, frequency, methodology, application
    - Secondary categories: data source, algorithm, performance metrics, risk measures
    """

    def __init__(self, config: AnalyzerWorkflowConfig):
        """Initialize analyzer workflow.

        Args:
            config: Analyzer workflow configuration
        """
        super().__init__(config)
        if not isinstance(config, AnalyzerWorkflowConfig):
            raise TypeError("config must be AnalyzerWorkflowConfig instance")
        self.config: AnalyzerWorkflowConfig = config

    def build_prompt(
        self, paper: Paper, tag_type: str = "primary", **kwargs
    ) -> str:
        """Build tag analysis prompt.

        Args:
            paper: Paper object containing content
            tag_type: Type of tags to generate ("primary" or "secondary")
            **kwargs: Additional context

        Returns:
            Formatted prompt string
        """
        # Determine categories based on tag type
        if tag_type == "primary":
            categories = self.config.primary_categories
            max_tags = min(self.config.max_tags, 5)
        else:
            categories = self.config.secondary_categories
            max_tags = min(self.config.max_tags, 8)

        # Prepare paper content
        content = ""
        if paper.full_text:
            content = paper.full_text[:4000]  # Limit for LLM context
        elif paper.abstract:
            content = paper.abstract

        prompt = f"""You are a quantitative finance research paper analyzer. Analyze this paper and generate {tag_type} tags in JSON format ONLY.

Paper Information:
Title: {paper.title}
Abstract: {paper.abstract}
Authors: {', '.join(paper.authors) if paper.authors else 'Not specified'}
{f"Content: {content}" if content else ""}

Task: Generate up to {max_tags} {tag_type} tags from these categories: {', '.join(categories)}

{tag_type.title()} Tag Categories:
"""

        # Add category-specific guidelines
        if tag_type == "primary":
            prompt += """- market_type: equity, fixed_income, forex, commodity, crypto, derivatives
- frequency: high_frequency, daily, weekly, monthly, quarterly, annual
- methodology: machine_learning, statistical_analysis, econometric, optimization, simulation
- application: portfolio_management, risk_management, trading_strategy, asset_pricing"""
        else:
            prompt += """- data_source: bloomberg, reuters, yahoo_finance, sec_filings, alternative_data
- algorithm: neural_networks, svm, random_forest, lstm, transformer, reinforcement_learning
- performance_metric: sharpe_ratio, sortino_ratio, max_drawdown, alpha, beta, information_ratio
- risk_measure: var, conditional_var, expected_shortfall, volatility, correlation"""

        prompt += f"""

Return the tags in this EXACT JSON format (no other text, just JSON):

{{
    "tags": [
        "market_type:equity",
        "methodology:machine_learning",
        "frequency:daily"
    ]
}}

Requirements:
- Analyze the paper's actual content and methodology
- Format as "category:value" strings
- Be specific and accurate based on paper content
- Focus on the most relevant and important aspects
- Return ONLY valid JSON, no explanations or additional text

Respond with JSON only:"""

        # Append custom instructions if configured
        return self._append_custom_instructions(prompt)

    def execute(self, paper: Paper, **kwargs) -> Tuple[List[str], List[str]]:
        """Execute analysis workflow to generate primary and secondary tags.

        Args:
            paper: Paper object to analyze
            **kwargs: Additional parameters

        Returns:
            Tuple of (primary_tag_strings, secondary_tag_strings)
        """
        logger.info(f"Analyzing paper for tags: {paper.title}")

        if not self.client:
            logger.warning("No LLM client available, returning empty tags")
            return [], []

        if not paper.full_text and not paper.abstract:
            logger.warning("No content available for tag analysis")
            return [], []

        try:
            primary_tags = []
            secondary_tags = []

            # Generate primary tags
            if self.config.enable_tag_analysis:
                logger.debug("Generating primary tags...")
                primary_tags = self._generate_tags(paper, "primary")

                logger.debug("Generating secondary tags...")
                secondary_tags = self._generate_tags(paper, "secondary")

            logger.info(
                f"Generated {len(primary_tags)} primary and "
                f"{len(secondary_tags)} secondary tags"
            )

            return primary_tags, secondary_tags

        except Exception as e:
            logger.error(f"Error in analysis workflow: {e}")
            return [], []

    def _generate_tags(self, paper: Paper, tag_type: str) -> List[str]:
        """Generate tags for specific type.

        Args:
            paper: Paper object
            tag_type: Type of tags ("primary" or "secondary")

        Returns:
            List of PaperTag objects
        """
        # Build prompt for this tag type
        prompt = self.build_prompt(paper, tag_type=tag_type)

        # Call LLM
        response = self._call_llm(prompt)
        if not response:
            logger.error(f"Failed to get LLM response for {tag_type} tags")
            return []

        # Parse JSON response
        try:
            return self._parse_tag_response(response)
        except Exception as e:
            logger.error(f"Failed to parse tag response for {tag_type}: {e}")
            return []

    def _parse_tag_response(self, response: str) -> List[str]:
        """Parse tag response from LLM.

        Args:
            response: Raw LLM response

        Returns:
            List of PaperTag objects
        """
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_content = response[json_start:json_end]
                tag_data = json.loads(json_content)
            else:
                tag_data = json.loads(response)

            # Parse tags as simple strings
            tags = []
            tag_list = tag_data.get("tags", [])

            if isinstance(tag_list, list):
                for tag_item in tag_list:
                    if isinstance(tag_item, str) and ":" in tag_item:
                        tags.append(tag_item)

            return tags

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tag response as JSON: {e}")
            logger.error(f"Response content: {response[:500]}...")
            return []

    def generate_methodology_summary(self, paper: Paper) -> Optional[str]:
        """Generate methodology summary from paper content.

        Args:
            paper: Paper object

        Returns:
            Methodology summary or None
        """
        if not paper.full_text:
            return None

        if not self.config.generate_methodology_summary:
            return None

        logger.debug("Generating methodology summary...")

        # Build methodology extraction prompt
        prompt = f"""Extract and summarize the methodology from this research paper.

Paper Title: {paper.title}
Abstract: {paper.abstract}
Content: {paper.full_text[:3000]}

Task: Generate a concise methodology summary (max {self.config.summary_max_length} characters) that covers:
- Main approach and framework used
- Key algorithms or techniques
- Data sources and preprocessing
- Model architecture or statistical methods

Return only the methodology summary, no other text:"""

        # Append custom instructions
        prompt = self._append_custom_instructions(prompt)

        # Call LLM
        response = self._call_llm(prompt)
        if response:
            # Truncate if too long
            if len(response) > self.config.summary_max_length:
                response = response[: self.config.summary_max_length] + "..."
            return response.strip()

        return None

    def generate_results_summary(self, paper: Paper) -> Optional[str]:
        """Generate results summary from paper content.

        Args:
            paper: Paper object

        Returns:
            Results summary or None
        """
        if not paper.full_text:
            return None

        if not self.config.generate_results_summary:
            return None

        logger.debug("Generating results summary...")

        # Build results extraction prompt
        prompt = f"""Extract and summarize the key results from this research paper.

Paper Title: {paper.title}
Abstract: {paper.abstract}
Content: {paper.full_text[:3000]}

Task: Generate a concise results summary (max {self.config.summary_max_length} characters) that covers:
- Main findings and outcomes
- Performance metrics and benchmarks
- Statistical significance or validation results
- Key insights and implications

Return only the results summary, no other text:"""

        # Append custom instructions
        prompt = self._append_custom_instructions(prompt)

        # Call LLM
        response = self._call_llm(prompt)
        if response:
            # Truncate if too long
            if len(response) > self.config.summary_max_length:
                response = response[: self.config.summary_max_length] + "..."
            return response.strip()

        return None

    def generate_tag_summary(
        self, primary_tags: List[str], secondary_tags: List[str]
    ) -> str:
        """Generate summary of generated tags.

        Args:
            primary_tags: Primary tags
            secondary_tags: Secondary tags

        Returns:
            Formatted tag summary
        """
        if not primary_tags and not secondary_tags:
            return "No tags generated."

        summary_parts = []

        if primary_tags:
            primary_summary = ", ".join(primary_tags)
            summary_parts.append(f"Primary: {primary_summary}")

        if secondary_tags:
            secondary_summary = ", ".join(secondary_tags)
            summary_parts.append(f"Secondary: {secondary_summary}")

        return " | ".join(summary_parts)
