"""Q&A generation flow for knowledge items."""

import json
from typing import List, Dict, Any, Optional

from quantmind.flow.base import BaseFlow
from quantmind.config.flows import QAFlowConfig
from quantmind.models.content import KnowledgeItem
from quantmind.models.analysis import QuestionAnswer
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class QAFlow(BaseFlow):
    """Flow for generating questions and answers from knowledge items.

    Uses LLM to generate insightful Q&A pairs covering different difficulty levels
    and categories including methodology, technical details, and critical analysis.
    Leverages the enhanced prompt engineering framework for flexible template-based prompts.
    """

    def __init__(self, config: QAFlowConfig):
        """Initialize Q&A flow.

        Args:
            config: Q&A flow configuration
        """
        super().__init__(config)
        if not isinstance(config, QAFlowConfig):
            raise TypeError("config must be QAFlowConfig instance")
        self.config: QAFlowConfig = config

    def execute(
        self,
        knowledge_item: KnowledgeItem,
        primary_tags: Optional[List] = None,
        secondary_tags: Optional[List] = None,
        **kwargs,
    ) -> List[QuestionAnswer]:
        """Execute Q&A generation flow.

        Args:
            knowledge_item: KnowledgeItem object to process
            primary_tags: Primary tags for additional context
            secondary_tags: Secondary tags for additional context
            **kwargs: Additional parameters

        Returns:
            List of generated QuestionAnswer objects
        """
        logger.info(f"Generating Q&A for content: {knowledge_item.title}")

        if not self.client:
            logger.warning("No LLM client available, returning empty Q&A")
            return []

        if not knowledge_item.content and not knowledge_item.abstract:
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
                    knowledge_item, difficulty, questions_per_difficulty
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
        self, knowledge_item: KnowledgeItem, difficulty: str, num_questions: int
    ) -> List[QuestionAnswer]:
        """Generate Q&A pairs for specific difficulty level.

        Args:
            knowledge_item: KnowledgeItem object
            difficulty: Difficulty level
            num_questions: Number of questions to generate

        Returns:
            List of QuestionAnswer objects
        """
        # Build prompt using the template system
        prompt = self.build_prompt(
            knowledge_item,
            difficulty=difficulty,
            question_categories=", ".join(self.config.question_categories),
            difficulty_levels=", ".join(self.config.difficulty_levels),
            num_questions=num_questions,
        )

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
        self,
        knowledge_item: KnowledgeItem,
        focus_area: str,
        num_questions: int = 3,
    ) -> List[QuestionAnswer]:
        """Generate focused Q&A for specific research area.

        Args:
            knowledge_item: KnowledgeItem object
            focus_area: Specific area to focus on
            num_questions: Number of questions to generate

        Returns:
            List of focused QuestionAnswer objects
        """
        logger.info(f"Generating focused Q&A for {focus_area}")

        if not self.client:
            return []

        # Build focused prompt using template system
        prompt = self.build_prompt(
            knowledge_item,
            focus_area=focus_area,
            num_questions=num_questions,
            specialized_focus=True,
        )

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
