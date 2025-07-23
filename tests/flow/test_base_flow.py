"""Simplified unit tests for BaseFlow class."""

import unittest
from typing import Any
from unittest.mock import MagicMock, Mock, patch

from quantmind.config.flows import BaseFlowConfig
from quantmind.config.llm import LLMConfig
from quantmind.flow.base import BaseFlow
from quantmind.models.content import KnowledgeItem


class TestBaseFlow(unittest.TestCase):
    """Test BaseFlow abstract class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test configuration
        self.llm_config = LLMConfig(
            model="gpt-4o",
            api_key="test-key",
            temperature=0.3,
            max_tokens=4000,
            custom_instructions="Test instructions",
        )

        self.config = BaseFlowConfig(
            llm_config=self.llm_config,
            prompt_template="{{system_prompt}}\n\nTitle: {{title}}\nContent: {{content}}\n\n{{custom_instructions}}",
            template_variables={"custom_var": "test_value"},
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
            def execute(self, knowledge_item, **kwargs):
                return {"result": "test"}

        self.TestFlow = TestFlow

    @patch("quantmind.flow.base.create_llm_block")
    def test_initialization_success(self, mock_create_llm_block):
        """Test successful BaseFlow initialization."""
        # Mock LLMBlock
        mock_llm_block = Mock()
        mock_create_llm_block.return_value = mock_llm_block

        # Initialize flow
        flow = self.TestFlow(self.config)

        # Verify initialization
        self.assertEqual(flow.config, self.config)
        self.assertEqual(flow.llm_block, mock_llm_block)
        mock_create_llm_block.assert_called_once_with(self.llm_config)

    @patch("quantmind.flow.base.create_llm_block")
    @patch("quantmind.flow.base.logger")
    def test_initialization_failure(self, mock_logger, mock_create_llm_block):
        """Test BaseFlow initialization when LLM creation fails."""
        # Mock LLM creation failure
        mock_create_llm_block.side_effect = Exception("Failed to create LLM")

        # Initialize flow
        flow = self.TestFlow(self.config)

        # Verify failure handling
        self.assertIsNone(flow.llm_block)
        mock_logger.error.assert_called_once()

    @patch("quantmind.flow.base.create_llm_block")
    def test_template_variable_extraction(self, mock_create_llm_block):
        """Test template variable extraction from KnowledgeItem."""
        mock_create_llm_block.return_value = Mock()
        flow = self.TestFlow(self.config)

        # Extract variables
        variables = flow.config.extract_template_variables(self.knowledge_item)

        # Verify core variables
        self.assertEqual(variables["title"], "Test Paper")
        self.assertEqual(
            variables["abstract"], "Test abstract about machine learning"
        )
        self.assertEqual(
            variables["content"],
            "This is test content about quantitative finance",
        )
        self.assertEqual(variables["authors"], "Test Author")
        self.assertEqual(variables["categories"], "q-fin.TR, cs.AI")
        self.assertEqual(variables["tags"], "machine learning, finance")
        self.assertEqual(variables["content_type"], "research_paper")
        self.assertEqual(variables["source"], "test")
        self.assertEqual(variables["custom_instructions"], "Test instructions")
        self.assertEqual(variables["custom_var"], "test_value")

    @patch("quantmind.flow.base.create_llm_block")
    def test_template_substitution(self, mock_create_llm_block):
        """Test template variable substitution."""
        mock_create_llm_block.return_value = Mock()
        flow = self.TestFlow(self.config)

        # Test template substitution
        template = (
            "Title: {{title}}\nMissing: {{missing_var}}\nCustom: {{custom_var}}"
        )
        variables = {"title": "Test Title", "custom_var": "test_value"}

        result = flow.config.substitute_template(template, variables)

        # Verify substitution
        self.assertIn("Title: Test Title", result)
        self.assertIn("Custom: test_value", result)
        self.assertIn("[missing_var: not available]", result)

    @patch("quantmind.flow.base.create_llm_block")
    def test_prompt_building_with_template(self, mock_create_llm_block):
        """Test prompt building using template system."""
        mock_create_llm_block.return_value = Mock()
        flow = self.TestFlow(self.config)

        # Build prompt
        prompt = flow.build_prompt(self.knowledge_item)

        # Verify prompt contains expected content
        self.assertIn("Test Paper", prompt)
        self.assertIn("This is test content", prompt)
        self.assertIn("Test instructions", prompt)

    @patch("quantmind.flow.base.create_llm_block")
    def test_prompt_building_fallback(self, mock_create_llm_block):
        """Test fallback prompt building functionality."""
        mock_create_llm_block.return_value = Mock()

        config = BaseFlowConfig(llm_config=self.llm_config)
        flow = self.TestFlow(config)

        # Test the fallback method directly
        fallback_prompt = flow._build_fallback_prompt(self.knowledge_item)

        # Verify fallback prompt contains basic info
        self.assertIn("Test Paper", fallback_prompt)
        self.assertIn("Test abstract", fallback_prompt)
        self.assertIn("research_paper", fallback_prompt)
        self.assertIn("Test instructions", fallback_prompt)

    @patch("quantmind.flow.base.create_llm_block")
    def test_custom_build_prompt_function(self, mock_create_llm_block):
        """Test custom prompt building function."""
        mock_create_llm_block.return_value = Mock()

        # Create custom prompt builder
        def custom_builder(knowledge_item, **kwargs):
            return f"Custom prompt for {knowledge_item.title}"

        # Create config with custom builder
        config_with_custom = BaseFlowConfig(
            llm_config=self.llm_config, custom_build_prompt=custom_builder
        )

        flow = self.TestFlow(config_with_custom)

        # Build prompt
        prompt = flow.build_prompt(self.knowledge_item)

        # Verify custom prompt was used
        self.assertEqual(prompt, "Custom prompt for Test Paper")

    @patch("quantmind.flow.base.create_llm_block")
    def test_template_validation(self, mock_create_llm_block):
        """Test template syntax validation."""
        mock_create_llm_block.return_value = Mock()

        # Test valid template
        valid_config = BaseFlowConfig(
            llm_config=self.llm_config,
            prompt_template="{{title}} - {{content}}",
        )
        flow = self.TestFlow(valid_config)
        is_valid, error = flow.validate_template()
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        # Test invalid template by setting it manually (bypass validation)
        invalid_config = BaseFlowConfig(llm_config=self.llm_config)
        invalid_config.prompt_template = "{{title}} - {{unclosed"
        flow = self.TestFlow(invalid_config)
        is_valid, error = flow.validate_template()
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)

    @patch("quantmind.flow.base.create_llm_block")
    def test_connection_testing(self, mock_create_llm_block):
        """Test LLM connection testing."""
        # Test with working connection
        mock_llm_block = Mock()
        mock_llm_block.test_connection.return_value = True
        mock_create_llm_block.return_value = mock_llm_block

        flow = self.TestFlow(self.config)
        self.assertTrue(flow.test_connection())

        # Test with failed connection
        mock_llm_block.test_connection.return_value = False
        self.assertFalse(flow.test_connection())

        # Test with no LLM block
        flow.llm_block = None
        self.assertFalse(flow.test_connection())

    @patch("quantmind.flow.base.create_llm_block")
    def test_get_model_info(self, mock_create_llm_block):
        """Test model information retrieval."""
        # Test with working LLM block
        mock_llm_block = Mock()
        mock_info = {"model": "gpt-4o", "provider": "openai"}
        mock_llm_block.get_info.return_value = mock_info
        mock_create_llm_block.return_value = mock_llm_block

        flow = self.TestFlow(self.config)
        info = flow.get_model_info()
        self.assertEqual(info, mock_info)

        # Test with no LLM block
        flow.llm_block = None
        info = flow.get_model_info()
        self.assertIn("error", info)

    @patch("quantmind.flow.base.create_llm_block")
    def test_llm_calling_methods(self, mock_create_llm_block):
        """Test LLM calling helper methods."""
        mock_llm_block = Mock()
        mock_llm_block.generate_text.return_value = "Test response"
        mock_create_llm_block.return_value = mock_llm_block

        flow = self.TestFlow(self.config)

        # Test text generation
        response = flow._call_llm("Test prompt")
        self.assertEqual(response, "Test response")
        mock_llm_block.generate_text.assert_called_with("Test prompt")

        # Test with no LLM block
        flow.llm_block = None
        self.assertIsNone(flow._call_llm("Test prompt"))

    @patch("quantmind.flow.base.create_llm_block")
    def test_llm_calling_with_exceptions(self, mock_create_llm_block):
        """Test LLM calling with exceptions."""
        mock_llm_block = Mock()
        mock_llm_block.generate_text.side_effect = Exception("API Error")
        mock_create_llm_block.return_value = mock_llm_block

        flow = self.TestFlow(self.config)

        # Test exception handling
        response = flow._call_llm("Test prompt")
        self.assertIsNone(response)

    @patch("quantmind.flow.base.create_llm_block")
    def test_client_property_backward_compatibility(
        self, mock_create_llm_block
    ):
        """Test client property for backward compatibility."""
        mock_llm_block = Mock()
        mock_create_llm_block.return_value = mock_llm_block

        flow = self.TestFlow(self.config)

        # Verify client property returns LLM block
        self.assertEqual(flow.client, mock_llm_block)

    @patch("quantmind.flow.base.create_llm_block")
    def test_prompt_preview_and_template_variables(self, mock_create_llm_block):
        """Test prompt preview and template variable debugging methods."""
        mock_create_llm_block.return_value = Mock()
        flow = self.TestFlow(self.config)

        # Test prompt preview
        preview = flow.get_prompt_preview(self.knowledge_item)
        self.assertIn("Test Paper", preview)

        # Test template variables preview
        variables = flow.get_template_variables(self.knowledge_item)
        self.assertIn("title", variables)
        self.assertIn("content", variables)
        self.assertEqual(variables["title"], "Test Paper")

    @patch("quantmind.flow.base.create_llm_block")
    def test_knowledge_item_with_meta_info(self, mock_create_llm_block):
        """Test handling of KnowledgeItem with meta_info."""
        mock_create_llm_block.return_value = Mock()
        flow = self.TestFlow(self.config)

        # Create knowledge item with meta_info
        knowledge_item_with_meta = KnowledgeItem(
            title="Test with Meta",
            content="Test content",
            meta_info={"doi": "10.1234/test", "journal": "Test Journal"},
        )

        # Extract variables
        variables = flow.config.extract_template_variables(
            knowledge_item_with_meta
        )

        # Verify meta_info variables are included with dot notation
        self.assertEqual(variables["meta_info.doi"], "10.1234/test")
        self.assertEqual(variables["meta_info.journal"], "Test Journal")


class TestBaseFlowConfig(unittest.TestCase):
    """Test BaseFlowConfig functionality."""

    def test_config_creation_with_create_method(self):
        """Test BaseFlowConfig creation using create() method."""
        config = BaseFlowConfig.create(
            model="gpt-4o",
            api_key="test-key",
            temperature=0.5,
            max_tokens=2000,
            custom_instructions="Test instructions",
        )

        # Verify LLM config composition
        self.assertEqual(config.llm_config.model, "gpt-4o")
        self.assertEqual(config.llm_config.api_key, "test-key")
        self.assertEqual(config.llm_config.temperature, 0.5)
        self.assertEqual(config.llm_config.max_tokens, 2000)
        self.assertEqual(
            config.llm_config.custom_instructions, "Test instructions"
        )

    def test_template_validation(self):
        """Test template validation in config."""
        # Valid template
        config = BaseFlowConfig(
            llm_config=LLMConfig(), prompt_template="{{title}} - {{content}}"
        )
        self.assertIsNotNone(config.prompt_template)

        # Invalid template should raise validation error
        with self.assertRaises(ValueError):
            BaseFlowConfig(
                llm_config=LLMConfig(),
                prompt_template="{{title} - {{content}}",  # Unbalanced braces
            )

    def test_default_system_prompt(self):
        """Test default system prompt behavior."""
        config = BaseFlowConfig(llm_config=LLMConfig())

        # Test class method
        default_prompt = config.get_default_system_prompt()
        self.assertIn("quantitative finance", default_prompt)

        # Test instance method
        system_prompt = config.get_system_prompt()
        self.assertIn("quantitative finance", system_prompt)


if __name__ == "__main__":
    unittest.main()
