"""Tests for the simplified workflow framework."""

import unittest
from unittest.mock import Mock, patch

from quantmind.workflow.base import BaseWorkflow
from quantmind.workflow.qa_workflow import QAWorkflow
from quantmind.config.workflows import (
    BaseWorkflowConfig,
    QAWorkflowConfig,
)
from quantmind.models.paper import Paper
from quantmind.models.analysis import QuestionAnswer


class TestBaseWorkflow(unittest.TestCase):
    """Test base workflow functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = BaseWorkflowConfig(
            llm_type="test", llm_name="test-model", api_key="test-key"
        )

        # Create concrete implementation for testing
        class TestWorkflow(BaseWorkflow):
            def build_prompt(self, paper, **kwargs):
                return f"Test prompt for {paper.title}"

            def execute(self, paper, **kwargs):
                return {"result": "test"}

        self.workflow = TestWorkflow(self.config)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = BaseWorkflowConfig(
            llm_type="openai", temperature=0.5, max_tokens=1000
        )
        self.assertEqual(config.llm_type, "openai")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 1000)

        # Invalid temperature
        with self.assertRaises(ValueError):
            BaseWorkflowConfig(temperature=3.0)  # > 2.0

    def test_custom_instructions(self):
        """Test custom instructions appending."""
        config = BaseWorkflowConfig(
            custom_instructions="Focus on trading strategies"
        )
        workflow = self.workflow.__class__(config)

        prompt = "Base prompt"
        result = workflow._append_custom_instructions(prompt)

        self.assertIn("Focus on trading strategies", result)
        self.assertIn("Base prompt", result)

    @patch("quantmind.workflow.base.logger")
    def test_client_initialization_failure(self, mock_logger):
        """Test handling of client initialization failure."""
        config = BaseWorkflowConfig(llm_type="unsupported")
        workflow = self.workflow.__class__(config)

        self.assertIsNone(workflow.client)
        mock_logger.warning.assert_called()


class TestQAWorkflow(unittest.TestCase):
    """Test Q&A workflow functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = QAWorkflowConfig(
            llm_type="test",
            num_questions=3,
            include_different_difficulties=True,
        )

        self.paper = Paper(
            title="Test Paper",
            abstract="Test abstract for machine learning in finance",
            authors=["Test Author"],
            source="test",
            full_text="Test content about machine learning and trading strategies.",
        )

        self.workflow = QAWorkflow(self.config)

    def test_build_prompt(self):
        """Test prompt building for Q&A."""
        prompt = self.workflow.build_prompt(
            self.paper, difficulty="intermediate"
        )

        self.assertIn("Test Paper", prompt)
        self.assertIn("Test abstract", prompt)
        self.assertIn("intermediate", prompt)
        self.assertIn("JSON", prompt)

    @patch.object(QAWorkflow, "_call_llm")
    def test_execute_success(self, mock_call_llm):
        """Test successful Q&A execution."""
        # Mock LLM response
        mock_response = """
        {
            "questions_answers": [
                {
                    "question": "What is the main methodology?",
                    "answer": "Machine learning approach",
                    "category": "methodology",
                    "difficulty": "intermediate",
                    "confidence": 0.9
                }
            ]
        }
        """
        mock_call_llm.return_value = mock_response

        # Mock client
        self.workflow.client = Mock()

        result = self.workflow.execute(self.paper)

        # Should generate questions for multiple difficulty levels
        self.assertGreater(len(result), 0)
        if result:
            self.assertIsInstance(result[0], QuestionAnswer)

    def test_execute_no_client(self):
        """Test execution with no LLM client."""
        self.workflow.client = None

        result = self.workflow.execute(self.paper)

        self.assertEqual(result, [])

    def test_execute_no_content(self):
        """Test execution with no paper content."""
        empty_paper = Paper(
            title="Empty Paper", authors=["Test"], source="test"
        )

        result = self.workflow.execute(empty_paper)

        self.assertEqual(result, [])

    def test_parse_qa_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        invalid_response = "This is not JSON"

        result = self.workflow._parse_qa_response(
            invalid_response, "intermediate"
        )

        self.assertEqual(result, [])

    def test_generate_summary(self):
        """Test Q&A summary generation."""
        qa_pairs = [
            QuestionAnswer(
                question="Q1",
                answer="A1",
                category="methodology",
                difficulty_level="beginner",
            ),
            QuestionAnswer(
                question="Q2",
                answer="A2",
                category="results",
                difficulty_level="intermediate",
            ),
        ]

        summary = self.workflow.generate_summary(qa_pairs)

        self.assertIn("methodology", summary)
        self.assertIn("results", summary)
        self.assertIn("1 questions", summary)  # Updated assertion


class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for workflow components."""

    def test_config_integration(self):
        """Test configuration integration across workflows."""
        # Test that configs can be created and used together
        base_config = BaseWorkflowConfig(llm_type="openai", temperature=0.5)

        qa_config = QAWorkflowConfig(
            llm_type="openai", temperature=0.3, num_questions=5
        )

        # Verify configurations
        self.assertEqual(qa_config.num_questions, 5)

    def test_paper_processing_pipeline(self):
        """Test end-to-end paper processing pipeline."""
        paper = Paper(
            title="Integration Test Paper",
            abstract="Testing the workflow pipeline",
            authors=["Test"],
            source="test",
        )

        # Test that workflows can be initialized
        qa_config = QAWorkflowConfig(llm_type="test")

        qa_workflow = QAWorkflow(qa_config)

        # Verify workflows can be initialized and have expected components
        self.assertEqual(qa_workflow.config.llm_type, "test")


if __name__ == "__main__":
    unittest.main()
