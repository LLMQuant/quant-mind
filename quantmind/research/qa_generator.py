"""LLM-based Q&A generator for research papers."""

import json
from typing import List, Optional, Dict, Any

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType
    from camel.agents import ChatAgent

    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False

from quantmind.models.paper import Paper
from quantmind.models.analysis import QuestionAnswer, AnalysisConfig
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class LLMQAGenerator:
    """LLM-based Q&A generator for research papers.

    Uses large language models to generate insightful questions and answers covering:
    - Theoretical understanding
    - Practical implementation
    - Critical analysis
    - Future directions
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize LLM Q&A generator.

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
            logger.info("Initialized CAMEL LLM client for Q&A generation")
        except Exception as e:
            logger.error(f"Failed to initialize CAMEL client: {e}")
            self.client = None

    def generate_questions_answers(
        self,
        paper: Paper,
        primary_tags: List = None,
        secondary_tags: List = None,
    ) -> List[QuestionAnswer]:
        """Generate questions and answers for the paper using LLM.

        Args:
            paper: Paper object to analyze
            primary_tags: Primary tags for context
            secondary_tags: Secondary tags for context

        Returns:
            List of generated question-answer pairs
        """
        logger.info(f"Generating Q&A for paper: {paper.title}")

        if not self.client:
            logger.warning("No LLM client available, returning empty Q&A")
            return []

        if not paper.full_text and not paper.abstract:
            logger.warning("No content available for Q&A generation")
            return []

        try:
            # Prepare context
            context = self._prepare_context(paper, primary_tags, secondary_tags)

            # Generate Q&A pairs
            qa_pairs = []

            if self.config.include_different_difficulties:
                difficulties = [
                    "beginner",
                    "intermediate",
                    "advanced",
                    "expert",
                ]
            else:
                difficulties = ["intermediate"]

            questions_per_difficulty = max(
                1, self.config.num_questions // len(difficulties)
            )

            for difficulty in difficulties:
                if len(qa_pairs) >= self.config.num_questions:
                    break

                # Generate Q&A for this difficulty level
                difficulty_qa = self._generate_difficulty_qa(
                    paper, context, difficulty, questions_per_difficulty
                )
                qa_pairs.extend(difficulty_qa)

            # Limit to requested number
            qa_pairs = qa_pairs[: self.config.num_questions]

            logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
            return qa_pairs

        except Exception as e:
            logger.error(f"Error in LLM Q&A generation: {e}")
            return []

    def _prepare_context(
        self,
        paper: Paper,
        primary_tags: List = None,
        secondary_tags: List = None,
    ) -> Dict[str, Any]:
        """Prepare context for Q&A generation.

        Args:
            paper: Paper object
            primary_tags: Primary tags
            secondary_tags: Secondary tags

        Returns:
            Context dictionary
        """
        context = {
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "published_date": paper.published_date,
            "source": paper.source,
        }

        # Add content (truncated for LLM)
        if paper.full_text:
            context["content"] = paper.full_text[:3000]  # Limit for LLM context
        elif paper.abstract:
            context["content"] = paper.abstract

        # Add tags
        if primary_tags:
            context["primary_tags"] = [tag.tag for tag in primary_tags]
        if secondary_tags:
            context["secondary_tags"] = [tag.tag for tag in secondary_tags]

        return context

    def _generate_difficulty_qa(
        self,
        paper: Paper,
        context: Dict[str, Any],
        difficulty: str,
        num_questions: int,
    ) -> List[QuestionAnswer]:
        """Generate Q&A pairs for a specific difficulty level.

        Args:
            paper: Paper object
            context: Context information
            difficulty: Difficulty level
            num_questions: Number of questions to generate

        Returns:
            List of Q&A pairs
        """
        prompt = self._build_qa_prompt(
            paper, context, difficulty, num_questions
        )

        try:
            response = self.client.step(prompt)
            response_content = response.msgs[0].content.strip()

            logger.debug(f"Raw LLM Q&A response: {response_content}")

            # Parse JSON response
            try:
                # Try to extract JSON from response if it's not pure JSON
                json_start = response_content.find("{")
                json_end = response_content.rfind("}") + 1

                if json_start != -1 and json_end > json_start:
                    json_content = response_content[json_start:json_end]
                    qa_data = json.loads(json_content)
                    return self._parse_qa_response(qa_data, difficulty)
                else:
                    # Try parsing the entire response as JSON
                    qa_data = json.loads(response_content)
                    return self._parse_qa_response(qa_data, difficulty)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM Q&A response as JSON: {e}")
                logger.error(f"Response content: {response_content[:500]}...")
                return []

        except Exception as e:
            logger.error(f"Error in LLM Q&A generation for {difficulty}: {e}")
            return []

    def _build_qa_prompt(
        self,
        paper: Paper,
        context: Dict[str, Any],
        difficulty: str,
        num_questions: int,
    ) -> str:
        """Build the Q&A generation prompt for the LLM.

        Args:
            paper: Paper object
            context: Context information
            difficulty: Difficulty level
            num_questions: Number of questions to generate

        Returns:
            Formatted prompt string
        """
        # Define question categories based on difficulty
        if difficulty == "beginner":
            categories = ["basic_understanding", "methodology_overview"]
        elif difficulty == "intermediate":
            categories = ["technical_details", "implementation", "analysis"]
        elif difficulty == "advanced":
            categories = [
                "theoretical_depth",
                "critical_analysis",
                "limitations",
            ]
        else:  # expert
            categories = [
                "theoretical_implications",
                "future_directions",
                "broader_impact",
            ]

        prompt = f"""You are a research paper Q&A generator. Generate {num_questions} questions and answers for this paper in JSON format ONLY.

Paper Information:
Title: {paper.title}
Abstract: {paper.abstract}
{f"Primary Tags: {', '.join(context.get('primary_tags', []))}" if context.get("primary_tags") else ""}
{f"Secondary Tags: {', '.join(context.get('secondary_tags', []))}" if context.get("secondary_tags") else ""}
{f"Content: {context.get('content', '')}" if context.get("content") else ""}

Difficulty Level: {difficulty}
Focus Categories: {", ".join(categories)}

Generate questions that are {difficulty} level appropriate and deeply insightful.

Return the Q&A in this EXACT JSON format (no other text, just JSON):

{{
    "questions_answers": [
        {{
            "question": "What is the main contribution of this research?",
            "answer": "The main contribution is...",
            "category": "methodology",
            "insight_level": "intermediate"
        }}
    ]
}}

Guidelines:
- Questions should be specific to this paper's content
- Answers should be comprehensive and educational
- Include practical insights and implementation considerations
- Address both strengths and limitations
- Provide actionable insights
- Focus on the most important aspects
- Return ONLY valid JSON, no explanations or additional text

Respond with JSON only:"""

        return prompt

    def _parse_qa_response(
        self, qa_data: Dict[str, Any], difficulty: str
    ) -> List[QuestionAnswer]:
        """Parse Q&A response from LLM.

        Args:
            qa_data: Q&A data from LLM
            difficulty: Difficulty level

        Returns:
            List of QuestionAnswer objects
        """
        qa_pairs = []

        qa_list = qa_data.get("questions_answers", [])
        if isinstance(qa_list, list):
            for qa_item in qa_list:
                if isinstance(qa_item, dict):
                    qa = QuestionAnswer(
                        question=qa_item.get("question", ""),
                        answer=qa_item.get("answer", ""),
                        difficulty=qa_item.get("difficulty", difficulty),
                        difficulty_level=qa_item.get("difficulty", difficulty),
                        category=qa_item.get("category", "general"),
                        confidence=qa_item.get("confidence", 0.8),
                    )
                    if qa.question and qa.answer:
                        qa_pairs.append(qa)

        return qa_pairs

    def _get_insight_level(self, difficulty: str) -> str:
        """Map difficulty level to insight level.

        Args:
            difficulty: Difficulty level

        Returns:
            Insight level
        """
        mapping = {
            "beginner": "basic",
            "intermediate": "intermediate",
            "advanced": "deep",
            "expert": "expert",
        }
        return mapping.get(difficulty, "intermediate")

    def generate_qa_summary(self, qa_pairs: List[QuestionAnswer]) -> str:
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

    def generate_focused_qa(
        self, paper: Paper, focus_area: str, num_questions: int = 3
    ) -> List[QuestionAnswer]:
        """Generate focused Q&A for a specific area.

        Args:
            paper: Paper object
            focus_area: Specific area to focus on (e.g., "methodology", "results", "implementation")
            num_questions: Number of questions to generate

        Returns:
            List of focused Q&A pairs
        """
        logger.info(f"Generating focused Q&A for {focus_area}")

        if not self.client:
            return []

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
            "insight_level": "deep"
        }}
    ]
}}"""

        try:
            response = self.client.step(prompt)
            response_content = response.msgs[0].content.strip()

            try:
                qa_data = json.loads(response_content)
                return self._parse_qa_response(qa_data, "advanced")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse focused Q&A response: {e}")
                return []

        except Exception as e:
            logger.error(f"Error in focused Q&A generation: {e}")
            return []

    def save_qa_to_file(
        self, qa_pairs: List[QuestionAnswer], output_path: str
    ) -> None:
        """Save Q&A pairs to file.

        Args:
            qa_pairs: List of Q&A pairs
            output_path: Output file path
        """
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        qa_data = []
        for qa in qa_pairs:
            qa_data.append(
                {
                    "question": qa.question,
                    "answer": qa.answer,
                    "category": qa.category,
                    "difficulty": qa.difficulty_level,
                    "insight_level": qa.insight_level,
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")
