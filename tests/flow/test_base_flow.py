"""Unit tests for BaseFlow class with new architecture."""

import unittest
from typing import Any
from unittest.mock import Mock, patch

from quantmind.config.flows import BaseFlowConfig
from quantmind.config.llm import LLMConfig
from quantmind.flow.base import BaseFlow
from quantmind.models.content import KnowledgeItem


class TestBaseFlow(unittest.TestCase):
    """Test BaseFlow abstract class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test configuration with multiple LLM blocks
        self.config = BaseFlowConfig(
            name="test_flow",
            llm_blocks={
                "primary_llm": LLMConfig(
                    model="gpt-4o",
                    api_key="test-key",
                    temperature=0.3,
                    max_tokens=4000,
                ),
                "secondary_llm": LLMConfig(
                    model="gpt-4o-mini", temperature=0.5, max_tokens=2000
                ),
            },
            prompt_templates={
                "test_template": "Hello {{ name }}, analyze {{ content }}",
                "summary_template": "Summarize: {{ text }}",
            },
        )

        # Create test knowledge item
        self.knowledge_item = KnowledgeItem(
            title="Test Paper",
            abstract="Test abstract about machine learning",
            content="This is test content about quantitative finance",
            authors=["Test Author"],
            categories=["q-fin.TR", "cs.AI"],
            tags=["machine learning", "finance"],
            content_type="research_paper",
            source="test",
        )

        # Create concrete implementation for testing
        class TestFlow(BaseFlow):
            def run(self, initial_input):
                return {"result": "test", "input": initial_input}

        self.TestFlow = TestFlow

    @patch("quantmind.flow.base.create_llm_block")
    def test_initialization_success(self, mock_create_llm_block):
        """Test successful BaseFlow initialization."""
        # Mock LLMBlock creation for each config
        mock_llm_block1 = Mock()
        mock_llm_block2 = Mock()
        mock_create_llm_block.side_effect = [mock_llm_block1, mock_llm_block2]

        # Initialize flow
        flow = self.TestFlow(self.config)

        # Verify initialization
        self.assertEqual(flow.config, self.config)
        self.assertEqual(len(flow._llm_blocks), 2)
        self.assertEqual(flow._llm_blocks["primary_llm"], mock_llm_block1)
        self.assertEqual(flow._llm_blocks["secondary_llm"], mock_llm_block2)
        self.assertEqual(len(flow._templates), 2)

    @patch("quantmind.flow.base.create_llm_block")
    @patch("quantmind.flow.base.logger")
    def test_initialization_failure(self, mock_logger, mock_create_llm_block):
        """Test BaseFlow initialization when LLM creation fails."""
        # Mock LLM creation failure for one block
        mock_llm_block = Mock()
        mock_create_llm_block.side_effect = [
            mock_llm_block,
            Exception("Failed to create LLM"),
        ]

        # Initialize flow
        flow = self.TestFlow(self.config)

        # Verify failure handling
        self.assertEqual(flow._llm_blocks["primary_llm"], mock_llm_block)
        self.assertIsNone(flow._llm_blocks["secondary_llm"])
        mock_logger.error.assert_called_once()

    @patch("quantmind.flow.base.create_llm_block")
    def test_direct_llm_block_access(self, mock_create_llm_block):
        """Test direct access to LLM blocks without wrapper methods."""
        mock_llm_block1 = Mock()
        mock_llm_block2 = Mock()
        mock_create_llm_block.side_effect = [mock_llm_block1, mock_llm_block2]

        flow = self.TestFlow(self.config)

        # Test direct access to existing LLM blocks
        self.assertEqual(flow._llm_blocks["primary_llm"], mock_llm_block1)
        self.assertEqual(flow._llm_blocks["secondary_llm"], mock_llm_block2)

        # Test KeyError for non-existent blocks
        with self.assertRaises(KeyError):
            _ = flow._llm_blocks["nonexistent"]

    @patch("quantmind.flow.base.create_llm_block")
    def test_failed_llm_initialization_handling(self, mock_create_llm_block):
        """Test handling of failed LLM block initialization."""
        mock_llm_block = Mock()
        mock_create_llm_block.side_effect = [
            mock_llm_block,
            Exception("Failed"),
        ]

        flow = self.TestFlow(self.config)

        # Successful initialization should work
        self.assertEqual(flow._llm_blocks["primary_llm"], mock_llm_block)

        # Failed initialization should be None
        self.assertIsNone(flow._llm_blocks["secondary_llm"])

    @patch("quantmind.flow.base.create_llm_block")
    def test_render_prompt_success(self, mock_create_llm_block):
        """Test rendering prompt with template successfully."""
        mock_create_llm_block.return_value = Mock()
        flow = self.TestFlow(self.config)

        result = flow._render_prompt(
            "test_template", name="Alice", content="test data"
        )

        self.assertEqual(result, "Hello Alice, analyze test data")

    @patch("quantmind.flow.base.create_llm_block")
    def test_render_prompt_template_not_found(self, mock_create_llm_block):
        """Test rendering prompt with non-existent template."""
        mock_create_llm_block.return_value = Mock()
        flow = self.TestFlow(self.config)

        with self.assertRaises(KeyError) as context:
            flow._render_prompt("nonexistent", name="Alice")

        self.assertIn(
            "Template 'nonexistent' not found", str(context.exception)
        )

    @patch("quantmind.flow.base.create_llm_block")
    def test_run_method_abstract(self, mock_create_llm_block):
        """Test that run method is properly implemented in concrete class."""
        mock_create_llm_block.return_value = Mock()
        flow = self.TestFlow(self.config)

        result = flow.run(self.knowledge_item)

        self.assertEqual(result["result"], "test")
        self.assertEqual(result["input"], self.knowledge_item)

    @patch("quantmind.flow.base.create_llm_block")
    def test_template_initialization(self, mock_create_llm_block):
        """Test that Jinja2 templates are properly initialized."""
        mock_create_llm_block.return_value = Mock()
        flow = self.TestFlow(self.config)

        # Verify templates are Jinja2 Template objects
        self.assertIn("test_template", flow._templates)
        self.assertIn("summary_template", flow._templates)

        # Test template rendering capabilities
        template = flow._templates["test_template"]
        result = template.render(name="Test", content="data")
        self.assertEqual(result, "Hello Test, analyze data")

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BaseFlow cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseFlow(self.config)


if __name__ == "__main__":
    unittest.main()
