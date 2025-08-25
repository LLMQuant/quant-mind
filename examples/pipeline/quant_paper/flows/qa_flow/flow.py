"""QA Flow for generating thoughtful questions from research papers."""

from datetime import datetime
from typing import Dict, List

from quantmind.config.flows import BaseFlowConfig
from quantmind.config.registry import register_flow_config
from quantmind.flow.base import BaseFlow
from quantmind.models.content import KnowledgeItem
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)

# JSON schema for structured question output
QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The refined question"},
        "category": {
            "type": "string",
            "enum": [
                "methodology",
                "findings",
                "implications",
                "applications",
                "theory",
            ],
            "description": "The category of the question",
        },
        "difficulty": {
            "type": "string",
            "enum": ["basic", "intermediate", "advanced"],
            "description": "The difficulty level of the question",
        },
    },
    "required": ["question", "category", "difficulty"],
    "additionalProperties": False,
}

# JSON schema for multiple questions generation
QUESTIONS_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "A thoughtful question about the research paper",
            },
            "description": "List of generated questions",
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


@register_flow_config("qa")
class QAFlowConfig(BaseFlowConfig):
    """Configuration for QA flow."""

    max_questions: int = 5
    question_depth: str = "deep"  # "basic", "intermediate", "deep"
    focus_areas: List[str] = ["methodology", "findings", "implications"]


class QAFlow(BaseFlow):
    """A flow that generates thoughtful QA questions from research papers.

    This flow analyzes the paper content and generates questions that promote
    deep thinking about the research, suitable for web display and engagement.
    """

    def run(self, document: KnowledgeItem) -> List[Dict[str, str]]:
        """Execute the QA generation flow.

        Args:
            document: KnowledgeItem (typically a Paper) to generate questions for

        Returns:
            List of dictionaries containing question data
        """
        logger.info(f"Starting QA flow for: {document.title}")

        content = document.content or ""
        if not content:
            logger.warning("No content available for QA generation")
            return []

        # Step 1: Generate initial questions
        initial_questions = self._generate_initial_questions(document)
        if not initial_questions:
            logger.error("Failed to generate initial questions")
            return []

        # Step 2: Refine and categorize questions
        refined_questions = self._refine_questions(initial_questions, document)

        # Step 3: Store in meta_info following best practice
        qa_metadata = {
            "questions": refined_questions,
            "generated_at": datetime.now().isoformat(),
            "model_used": self.config.llm_blocks["question_generator"].model,
            "question_count": len(refined_questions),
            "focus_areas": self.config.focus_areas,
            "depth_level": self.config.question_depth,
        }

        document.meta_info["qa_data"] = qa_metadata
        logger.info(
            f"Generated {len(refined_questions)} questions for: {document.title}"
        )

        return refined_questions

    def _generate_initial_questions(self, document: KnowledgeItem) -> List[str]:
        """Generate initial set of questions based on paper content."""
        generator_llm = self._llm_blocks["question_generator"]

        # Prepare content for analysis (limit size for API efficiency)
        analysis_content = self._prepare_content_for_analysis(document)

        prompt = self._render_prompt(
            "generate_questions_template",
            title=document.title,
            abstract=document.abstract or "",
            content=analysis_content,
            max_questions=self.config.max_questions,
            depth=self.config.question_depth,
            focus_areas=", ".join(self.config.focus_areas),
        )

        try:
            # Use structured output generation for initial questions
            response_format = {
                "type": "json_object",
                "response_schema": QUESTIONS_LIST_SCHEMA,
            }
            structured_response = generator_llm.generate_structured_output(
                prompt, response_format=response_format
            )

            if structured_response and isinstance(structured_response, dict):
                questions = structured_response.get("questions", [])
                # Limit to max_questions and ensure all are strings ending with '?'
                questions = [
                    q
                    for q in questions
                    if isinstance(q, str) and q.strip().endswith("?")
                ]
                questions = questions[: self.config.max_questions]
                logger.debug(f"Generated {len(questions)} initial questions")
                return questions
            else:
                logger.warning(
                    "No structured response received for initial questions"
                )
                return []

        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []

    def _refine_questions(
        self, questions: List[str], document: KnowledgeItem
    ) -> List[Dict[str, str]]:
        """Refine questions and add metadata using structured output."""
        refiner_llm = self._llm_blocks["question_refiner"]
        refined_questions = []

        for i, question in enumerate(questions):
            try:
                refine_prompt = self._render_prompt(
                    "refine_question_template",
                    question=question,
                    title=document.title,
                    abstract=document.abstract or "",
                )

                # Use structured output generation with schema
                response_format = {
                    "type": "json_object",
                    "response_schema": QUESTION_SCHEMA,
                }
                structured_response = refiner_llm.generate_structured_output(
                    refine_prompt, response_format=response_format
                )

                if structured_response and isinstance(
                    structured_response, dict
                ):
                    # Validate required fields and add ID
                    question_data = {
                        "id": f"q_{i + 1}",
                        "question": structured_response.get(
                            "question", question
                        ),
                        "category": structured_response.get(
                            "category", "general"
                        ),
                        "difficulty": structured_response.get(
                            "difficulty", "intermediate"
                        ),
                    }
                    refined_questions.append(question_data)
                else:
                    # Fallback to original question
                    refined_questions.append(
                        {
                            "id": f"q_{i + 1}",
                            "question": question,
                            "category": "general",
                            "difficulty": "intermediate",
                        }
                    )

            except Exception as e:
                logger.warning(f"Error refining question {i + 1}: {e}")
                # Fallback to original question
                refined_questions.append(
                    {
                        "id": f"q_{i + 1}",
                        "question": question,
                        "category": "general",
                        "difficulty": "intermediate",
                    }
                )

        return refined_questions

    def _prepare_content_for_analysis(self, document: KnowledgeItem) -> str:
        """Prepare content for analysis, limiting size for API efficiency."""
        content = document.content or ""

        # Limit content to avoid API token limits
        max_chars = 8000  # Approximately 2000 tokens
        if len(content) > max_chars:
            # Take first portion + last portion to capture intro and conclusion
            first_half = content[: max_chars // 2]
            last_half = content[-max_chars // 2 :]
            content = (
                first_half + "\n\n[... content truncated ...]\n\n" + last_half
            )

        return content
