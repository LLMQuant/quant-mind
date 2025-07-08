"""Paper summary generation workflow for QuantMind framework."""

import json
from typing import Any, Dict, List, Optional

from quantmind.models.paper import Paper
from quantmind.workflow.base import BaseWorkflow
from quantmind.config.workflows import PaperSummaryWorkflowConfig
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class PaperSummaryWorkflow(BaseWorkflow):
    """Workflow for generating comprehensive paper summaries using LLM.

    This workflow analyzes research papers and generates structured summaries
    focusing on key findings, methodology, results, and implications.
    """

    def __init__(self, config: PaperSummaryWorkflowConfig):
        """Initialize paper summary workflow.

        Args:
            config: Configuration for summary generation
        """
        super().__init__(config)
        self.config = config

    def build_prompt(self, paper: Paper, **kwargs) -> str:
        """Build prompt for paper summary generation.

        Args:
            paper: Paper object containing content
            **kwargs: Additional context or parameters

        Returns:
            Formatted prompt string for summary generation
        """
        # Base prompt template
        prompt = f"""You are an expert research analyst specializing in quantitative finance and machine learning.
Please analyze the following research paper and generate a {self.config.summary_type} summary.

Paper Information:
- Title: {paper.title}
- Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}
- Abstract: {paper.abstract}
- Categories: {', '.join(paper.categories) if paper.categories else 'None'}
- Tags: {', '.join(paper.tags) if paper.tags else 'None'}

Full Text Content:
{paper.full_text if paper.full_text else paper.abstract}

Please generate a summary with the following requirements:
- Summary type: {self.config.summary_type}
- Maximum length: {self.config.max_summary_length} words
- Focus on quantitative aspects: {self.config.focus_on_quantitative_aspects}
- Include key findings: {self.config.include_key_findings}
- Include methodology: {self.config.include_methodology}
- Include results: {self.config.include_results}
- Include implications: {self.config.include_implications}
- Highlight innovations: {self.config.highlight_innovations}
- Include limitations: {self.config.include_limitations}

Output format: {self.config.output_format}

Please provide a clear, well-structured summary that captures the essence of the research."""

        # Append custom instructions if provided
        prompt = self._append_custom_instructions(prompt)

        return prompt

    def execute(self, paper: Paper, **kwargs) -> Dict[str, Any]:
        """Execute paper summary workflow.

        Args:
            paper: Paper object to summarize
            **kwargs: Additional parameters

        Returns:
            Dictionary containing summary results
        """
        logger.info(f"Generating summary for paper: {paper.title}")

        # Build prompt
        prompt = self.build_prompt(paper, **kwargs)

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
                    "paper_id": paper.get_primary_id(),
                    "paper_title": paper.title,
                    "summary_type": self.config.summary_type,
                    "output_format": self.config.output_format,
                    "word_count": len(response.split()),
                    "max_length": self.config.max_summary_length,
                }
            )

            logger.info(f"Successfully generated summary for {paper.title}")
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

    def generate_brief_summary(self, paper: Paper) -> str:
        """Generate a brief summary (convenience method).

        Args:
            paper: Paper to summarize

        Returns:
            Brief summary text
        """
        # Temporarily modify config for brief summary
        original_type = self.config.summary_type
        original_length = self.config.max_summary_length

        self.config.summary_type = "brief"
        self.config.max_summary_length = 200

        try:
            result = self.execute(paper)
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

    def generate_executive_summary(self, paper: Paper) -> Dict[str, Any]:
        """Generate an executive summary (convenience method).

        Args:
            paper: Paper to summarize

        Returns:
            Executive summary data
        """
        # Temporarily modify config for executive summary
        original_type = self.config.summary_type
        original_length = self.config.max_summary_length

        self.config.summary_type = "executive"
        self.config.max_summary_length = 500

        try:
            result = self.execute(paper)

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
