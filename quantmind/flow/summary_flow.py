"""Content summary generation flow for QuantMind framework."""

import json
from typing import Any, Dict, List, Optional

from quantmind.models.content import KnowledgeItem
from quantmind.flow.base import BaseFlow
from quantmind.config.flows import SummaryFlowConfig
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class SummaryFlow(BaseFlow):
    """Flow for generating comprehensive content summaries using LLM.

    This flow analyzes knowledge items and generates structured summaries
    focusing on key findings, methodology, results, and implications.
    Leverages the enhanced prompt engineering framework for flexible templates.
    """

    def __init__(self, config: SummaryFlowConfig):
        """Initialize content summary flow.

        Args:
            config: Configuration for summary generation
        """
        super().__init__(config)
        self.config = config

    def execute(
        self, knowledge_item: KnowledgeItem, **kwargs
    ) -> Dict[str, Any]:
        """Execute content summary flow.

        Args:
            knowledge_item: KnowledgeItem object to summarize
            **kwargs: Additional parameters

        Returns:
            Dictionary containing summary results
        """
        logger.info(f"Generating summary for content: {knowledge_item.title}")

        # Build prompt using template system
        prompt = self.build_prompt(knowledge_item, **kwargs)

        # Call LLM
        response = self._call_llm(prompt)

        if not response:
            logger.error("Failed to generate summary - no LLM response")
            return self._create_error_result("Failed to generate summary")

        # Parse response based on output format
        try:
            if self.config.output_format == "structured":
                summary_data = self._parse_structured_response(response)
            elif self.config.output_format == "bullet_points":
                summary_data = self._parse_bullet_points_response(response)
            else:  # narrative
                summary_data = self._parse_narrative_response(response)

            # Add metadata
            summary_data.update(
                {
                    "content_id": knowledge_item.get_primary_id(),
                    "content_title": knowledge_item.title,
                    "content_type": getattr(
                        knowledge_item, "content_type", "generic"
                    ),
                    "summary_type": self.config.summary_type,
                    "output_format": self.config.output_format,
                    "word_count": len(response.split()),
                    "max_length": self.config.max_summary_length,
                }
            )

            logger.info(
                f"Successfully generated summary for {knowledge_item.title}"
            )
            return summary_data

        except Exception as e:
            logger.error(f"Failed to parse summary response: {e}")
            return self._create_error_result(
                f"Failed to parse summary: {str(e)}"
            )

    def _parse_structured_response(self, response: str) -> Dict[str, Any]:
        """Parse structured JSON response from LLM.

        Args:
            response: LLM response string

        Returns:
            Parsed summary data
        """
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()

            data = json.loads(json_str)

            # Ensure required fields exist
            required_fields = [
                "summary",
                "key_findings",
                "methodology",
                "results",
            ]
            for field in required_fields:
                if field not in data:
                    data[field] = ""

            return data

        except json.JSONDecodeError:
            # Fallback to narrative parsing if JSON parsing fails
            logger.warning(
                "Failed to parse JSON response, falling back to narrative parsing"
            )
            return self._parse_narrative_response(response)

    def _parse_narrative_response(self, response: str) -> Dict[str, Any]:
        """Parse narrative text response from LLM.

        Args:
            response: LLM response string

        Returns:
            Parsed summary data
        """
        # Clean up response
        response = response.strip()

        # Try to extract sections if they exist
        sections = {
            "summary": response,
            "key_findings": "",
            "methodology": "",
            "results": "",
            "implications": "",
            "limitations": "",
        }

        # Look for common section headers
        section_headers = {
            "key_findings": ["Key Findings:", "Main Findings:", "Findings:"],
            "methodology": ["Methodology:", "Methods:", "Approach:"],
            "results": ["Results:", "Findings:", "Outcomes:"],
            "implications": ["Implications:", "Impact:", "Applications:"],
            "limitations": ["Limitations:", "Caveats:", "Constraints:"],
        }

        for section, headers in section_headers.items():
            for header in headers:
                if header in response:
                    # Extract section content
                    start_idx = response.find(header) + len(header)
                    end_idx = len(response)

                    # Find next section
                    for next_header in [
                        h
                        for headers_list in section_headers.values()
                        for h in headers_list
                    ]:
                        next_idx = response.find(next_header, start_idx)
                        if next_idx != -1 and next_idx < end_idx:
                            end_idx = next_idx

                    sections[section] = response[start_idx:end_idx].strip()
                    break

        return sections

    def _parse_bullet_points_response(self, response: str) -> Dict[str, Any]:
        """Parse bullet points response from LLM.

        Args:
            response: LLM response string

        Returns:
            Parsed summary data
        """
        # Clean up response
        response = response.strip()

        # Split into lines and extract bullet points
        lines = response.split("\n")
        bullet_points = []

        for line in lines:
            line = line.strip()
            if line.startswith(("-", "•", "*", "→", "▶")):
                bullet_points.append(line[1:].strip())
            elif line and not line.startswith("#"):  # Skip headers
                bullet_points.append(line)

        return {
            "summary": response,
            "bullet_points": bullet_points,
            "key_findings": "",
            "methodology": "",
            "results": "",
            "implications": "",
            "limitations": "",
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure.

        Args:
            error_message: Error description

        Returns:
            Error result dictionary
        """
        return {
            "error": True,
            "error_message": error_message,
            "summary": "",
            "key_findings": "",
            "methodology": "",
            "results": "",
            "implications": "",
            "limitations": "",
        }

    def generate_brief_summary(self, knowledge_item: KnowledgeItem) -> str:
        """Generate a brief summary (convenience method).

        Args:
            knowledge_item: KnowledgeItem to summarize

        Returns:
            Brief summary text
        """
        # Temporarily modify config for brief summary
        original_type = self.config.summary_type
        original_length = self.config.max_summary_length

        self.config.summary_type = "brief"
        self.config.max_summary_length = 200

        try:
            result = self.execute(knowledge_item)
            summary = result.get("summary", "")

            # Restore original config
            self.config.summary_type = original_type
            self.config.max_summary_length = original_length

            return summary
        except Exception as e:
            logger.error(f"Failed to generate brief summary: {e}")
            # Restore original config
            self.config.summary_type = original_type
            self.config.max_summary_length = original_length
            return ""

    def generate_executive_summary(
        self, knowledge_item: KnowledgeItem
    ) -> Dict[str, Any]:
        """Generate an executive summary (convenience method).

        Args:
            knowledge_item: KnowledgeItem to summarize

        Returns:
            Executive summary data
        """
        # Temporarily modify config for executive summary
        original_type = self.config.summary_type
        original_length = self.config.max_summary_length

        self.config.summary_type = "executive"
        self.config.max_summary_length = 500

        try:
            result = self.execute(knowledge_item)

            # Restore original config
            self.config.summary_type = original_type
            self.config.max_summary_length = original_length

            return result
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            # Restore original config
            self.config.summary_type = original_type
            self.config.max_summary_length = original_length
            return self._create_error_result(
                f"Failed to generate executive summary: {str(e)}"
            )
