"""Q&A generation workflow for research papers."""

import json
from typing import List, Dict, Any, Optional

from quantmind.workflow.base import BaseWorkflow
from quantmind.config.workflows import QAWorkflowConfig
from quantmind.models.paper import Paper
from quantmind.models.analysis import QuestionAnswer
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class QAWorkflow(BaseWorkflow):
    """Workflow for generating questions and answers from research papers.

    Uses LLM to generate insightful Q&A pairs covering different difficulty levels
    and categories including methodology, technical details, and critical analysis.
    """

    def __init__(self, config: QAWorkflowConfig):
        """Initialize Q&A workflow.

        Args:
            config: Q&A workflow configuration
        """
        super().__init__(config)
        if not isinstance(config, QAWorkflowConfig):
            raise TypeError("config must be QAWorkflowConfig instance")
        self.config: QAWorkflowConfig = config

    def build_prompt(
        self,
        paper: Paper,
        difficulty: str = "intermediate",
        categories: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Build Q&A generation prompt.

        Args:
            paper: Paper object containing content
            difficulty: Difficulty level for questions
            categories: Specific categories to focus on
            **kwargs: Additional context

        Returns:
            Formatted prompt string
        """
        # Use provided categories or default from config
        if categories is None:
            categories = self.config.question_categories

        # Prepare paper content (truncated for LLM context)
        content = ""
        if paper.full_text:
            content = paper.full_text[:3000]  # Limit for LLM context
        elif paper.abstract:
            content = paper.abstract

        # Build base prompt
        prompt = f"""You are a research paper Q&A generator. Generate questions and answers for this paper in JSON format ONLY.

Paper Information:
Title: {paper.title}
Abstract: {paper.abstract}
Authors: {', '.join(paper.authors) if paper.authors else 'Not specified'}
{f"Content: {content}" if content else ""}

Task: Generate {self.config.num_questions} questions and answers at {difficulty} difficulty level.

Focus Categories: {', '.join(categories)}

Guidelines for {difficulty} level:
"""

        # Add difficulty-specific guidelines
        if difficulty == "beginner":
            prompt += """- Basic understanding and concept clarification
- Overview of methodology and approach
- Simple explanations of key terms
- General context and background"""
        elif difficulty == "intermediate":
            prompt += """- Technical details and implementation
- Analysis of methodology and results
- Comparison with existing approaches
- Practical applications and implications"""
        elif difficulty == "advanced":
            prompt += """- Theoretical depth and mathematical foundations
- Critical analysis of limitations and assumptions
- Novel contributions and innovations
- Complex relationships and dependencies"""
        else:  # expert
            prompt += """- Theoretical implications and future research directions
- Broader impact on the field
- Integration with cutting-edge research
- Advanced methodological considerations"""

        prompt += f"""

Return the Q&A in this EXACT JSON format (no other text, just JSON):

{{
    "questions_answers": [
        {{
            "question": "What is the main contribution of this research?",
            "answer": "The main contribution is...",
            "category": "methodology",
            "difficulty": "{difficulty}",
            "confidence": 0.9
        }}
    ]
}}

Requirements:
- Questions should be specific to this paper's content
- Answers should be comprehensive and educational
- Include practical insights and implementation considerations
- Address both strengths and limitations
- Focus on the most important aspects
- Return ONLY valid JSON, no explanations or additional text

Respond with JSON only:"""

        # Append custom instructions if configured
        return self._append_custom_instructions(prompt)

    def execute(
        self,
        paper: Paper,
        primary_tags: Optional[List] = None,
        secondary_tags: Optional[List] = None,
        **kwargs,
    ) -> List[QuestionAnswer]:
        """Execute Q&A generation workflow.

        Args:
            paper: Paper object to process
            primary_tags: Primary tags for additional context
            secondary_tags: Secondary tags for additional context
            **kwargs: Additional parameters

        Returns:
            List of generated QuestionAnswer objects
        """
        logger.info(f"Generating Q&A for paper: {paper.title}")

        if not self.client:
            logger.warning("No LLM client available, returning empty Q&A")
            return []

        if not paper.full_text and not paper.abstract:
            logger.warning("No content available for Q&A generation")
            return []

        try:
            qa_pairs = []

            # Determine difficulty levels to generate
            if self.config.include_different_difficulties:
                difficulties = self.config.difficulty_levels
                questions_per_difficulty = max(
                    1, self.config.num_questions // len(difficulties)
                )
            else:
                difficulties = ["intermediate"]
                questions_per_difficulty = self.config.num_questions

            # Generate Q&A for each difficulty level
            for difficulty in difficulties:
                if len(qa_pairs) >= self.config.num_questions:
                    break

                logger.debug(f"Generating {difficulty} level questions...")
                difficulty_qa = self._generate_difficulty_qa(
                    paper, difficulty, questions_per_difficulty
                )
                qa_pairs.extend(difficulty_qa)

            # Limit to requested number
            qa_pairs = qa_pairs[: self.config.num_questions]

            logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
            return qa_pairs

        except Exception as e:
            logger.error(f"Error in Q&A generation workflow: {e}")
            return []

    def _generate_difficulty_qa(
        self, paper: Paper, difficulty: str, num_questions: int
    ) -> List[QuestionAnswer]:
        """Generate Q&A pairs for specific difficulty level.

        Args:
            paper: Paper object
            difficulty: Difficulty level
            num_questions: Number of questions to generate

        Returns:
            List of QuestionAnswer objects
        """
        # Build prompt for this difficulty level
        prompt = self.build_prompt(paper, difficulty=difficulty)

        # Call LLM
        response = self._call_llm(prompt)
        if not response:
            logger.error(
                f"Failed to get LLM response for {difficulty} questions"
            )
            return []

        # Parse JSON response
        try:
            return self._parse_qa_response(response, difficulty)
        except Exception as e:
            logger.error(f"Failed to parse Q&A response for {difficulty}: {e}")
            return []

    def _parse_qa_response(
        self, response: str, difficulty: str
    ) -> List[QuestionAnswer]:
        """Parse Q&A response from LLM.

        Args:
            response: Raw LLM response
            difficulty: Difficulty level

        Returns:
            List of QuestionAnswer objects
        """
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_content = response[json_start:json_end]
                qa_data = json.loads(json_content)
            else:
                qa_data = json.loads(response)

            # Parse question-answer pairs
            qa_pairs = []
            qa_list = qa_data.get("questions_answers", [])

            if isinstance(qa_list, list):
                for qa_item in qa_list:
                    if isinstance(qa_item, dict):
                        qa = QuestionAnswer(
                            question=qa_item.get("question", ""),
                            answer=qa_item.get("answer", ""),
                            difficulty=qa_item.get("difficulty", difficulty),
                            difficulty_level=qa_item.get(
                                "difficulty", difficulty
                            ),
                            category=qa_item.get("category", "general"),
                            confidence=qa_item.get("confidence", 0.8),
                        )
                        if qa.question and qa.answer:
                            qa_pairs.append(qa)

            return qa_pairs

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Q&A response as JSON: {e}")
            logger.error(f"Response content: {response[:500]}...")
            return []

    def generate_focused_qa(
        self, paper: Paper, focus_area: str, num_questions: int = 3
    ) -> List[QuestionAnswer]:
        """Generate focused Q&A for specific research area.

        Args:
            paper: Paper object
            focus_area: Specific area to focus on
            num_questions: Number of questions to generate

        Returns:
            List of focused QuestionAnswer objects
        """
        logger.info(f"Generating focused Q&A for {focus_area}")

        if not self.client:
            return []

        # Build focused prompt
        prompt = f"""Generate {num_questions} focused questions and answers about the {focus_area} of this research paper.

Paper Information:
Title: {paper.title}
Abstract: {paper.abstract}
{f"Content: {paper.full_text[:2000]}..." if paper.full_text else ""}

Focus Area: {focus_area}

Generate questions that specifically address:
- Key aspects of the {focus_area}
- Practical implications
- Potential improvements or extensions
- Critical analysis of the {focus_area}

Return the Q&A in this exact JSON format:
{{
    "questions_answers": [
        {{
            "question": "string",
            "answer": "string",
            "category": "{focus_area}",
            "difficulty": "advanced",
            "confidence": 0.8
        }}
    ]
}}"""

        # Append custom instructions
        prompt = self._append_custom_instructions(prompt)

        # Call LLM and parse response
        response = self._call_llm(prompt)
        if response:
            return self._parse_qa_response(response, "advanced")

        return []

    def generate_summary(self, qa_pairs: List[QuestionAnswer]) -> str:
        """Generate summary of Q&A pairs.

        Args:
            qa_pairs: List of Q&A pairs

        Returns:
            Formatted Q&A summary
        """
        if not qa_pairs:
            return "No Q&A pairs generated."

        # Group by category
        categories = {}
        for qa in qa_pairs:
            if qa.category not in categories:
                categories[qa.category] = []
            categories[qa.category].append(qa)

        summary_parts = []
        for category, qa_list in categories.items():
            difficulties = [qa.difficulty_level for qa in qa_list]
            summary_parts.append(
                f"{category}: {len(qa_list)} questions ({', '.join(set(difficulties))})"
            )

        return " | ".join(summary_parts)
