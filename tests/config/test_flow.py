"""Tests for flow configuration."""

import unittest
from unittest.mock import patch

from quantmind.config.flows import (
    BaseFlowConfig,
    SummaryFlowConfig,
    ChunkingStrategy,
)
from quantmind.config.llm import LLMConfig


class TestBaseFlowConfig(unittest.TestCase):
    """Test cases for BaseFlowConfig."""

    def test_init_basic(self):
        """Test basic initialization."""
        config = BaseFlowConfig(name="test_flow")

        self.assertEqual(config.name, "test_flow")
        self.assertEqual(config.llm_blocks, {})
        self.assertEqual(config.prompt_templates, {})
        self.assertIsNone(config.prompt_templates_path)

    def test_direct_llm_block_assignment(self):
        """Test direct assignment of LLM blocks."""
        config = BaseFlowConfig(name="test_flow")
        llm_config = LLMConfig(model="gpt-4o", temperature=0.5)

        config.llm_blocks["test_llm"] = llm_config

        self.assertIn("test_llm", config.llm_blocks)
        self.assertEqual(config.llm_blocks["test_llm"], llm_config)

    def test_direct_prompt_template_assignment(self):
        """Test direct assignment of prompt templates."""
        config = BaseFlowConfig(name="test_flow")
        template = "Hello {{ name }}, how are you?"

        config.prompt_templates["greeting"] = template

        self.assertIn("greeting", config.prompt_templates)
        self.assertEqual(config.prompt_templates["greeting"], template)

    def test_config_initialization_with_resources(self):
        """Test initialization with resources."""
        llm_config = LLMConfig(model="gpt-4o")
        template = "Test template"

        config = BaseFlowConfig(
            name="test_flow",
            llm_blocks={"test_llm": llm_config},
            prompt_templates={"test": template},
        )

        self.assertEqual(config.llm_blocks["test_llm"], llm_config)
        self.assertEqual(config.prompt_templates["test"], template)

    def test_empty_config(self):
        """Test accessing non-existent items raises KeyError."""
        config = BaseFlowConfig(name="test_flow")

        with self.assertRaises(KeyError):
            _ = config.llm_blocks["nonexistent"]

        with self.assertRaises(KeyError):
            _ = config.prompt_templates["nonexistent"]


class TestSummaryFlowConfig(unittest.TestCase):
    """Test cases for SummaryFlowConfig."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = SummaryFlowConfig(name="summary_flow")

        self.assertEqual(config.name, "summary_flow")
        self.assertEqual(config.chunk_size, 2000)
        self.assertEqual(config.use_chunking, True)
        self.assertEqual(config.chunk_strategy, ChunkingStrategy.BY_SIZE)
        self.assertIsNone(config.chunk_custom_strategy)

        # Check default LLM blocks are created
        self.assertIn("cheap_summarizer", config.llm_blocks)
        self.assertIn("powerful_combiner", config.llm_blocks)

        # Check default templates are created
        self.assertIn("summarize_chunk_template", config.prompt_templates)
        self.assertIn("combine_summaries_template", config.prompt_templates)

    def test_init_with_custom_chunk_size(self):
        """Test initialization with custom chunk size."""
        config = SummaryFlowConfig(name="summary_flow", chunk_size=1000)

        self.assertEqual(config.chunk_size, 1000)

    def test_default_llm_blocks_configuration(self):
        """Test default LLM block configurations."""
        config = SummaryFlowConfig(name="summary_flow")

        cheap_config = config.llm_blocks["cheap_summarizer"]
        self.assertEqual(cheap_config.model, "gpt-4o-mini")
        self.assertEqual(cheap_config.temperature, 0.3)
        self.assertEqual(cheap_config.max_tokens, 1000)

        powerful_config = config.llm_blocks["powerful_combiner"]
        self.assertEqual(powerful_config.model, "gpt-4o")
        self.assertEqual(powerful_config.temperature, 0.3)
        self.assertEqual(powerful_config.max_tokens, 2000)

    def test_default_prompt_templates(self):
        """Test default prompt templates are properly set."""
        config = SummaryFlowConfig(name="summary_flow")

        chunk_template = config.prompt_templates["summarize_chunk_template"]
        self.assertIn("chunk_text", chunk_template)
        self.assertIn("financial research expert", chunk_template.lower())

        combine_template = config.prompt_templates["combine_summaries_template"]
        self.assertIn("summaries", combine_template)
        self.assertIn("coherent", combine_template.lower())

    def test_custom_llm_blocks_preserved(self):
        """Test that custom LLM blocks are preserved."""
        custom_llm_blocks = {
            "custom_llm": LLMConfig(model="custom-model", temperature=0.7)
        }

        config = SummaryFlowConfig(
            name="summary_flow", llm_blocks=custom_llm_blocks
        )

        # Custom blocks should be preserved, defaults not added
        self.assertEqual(len(config.llm_blocks), 1)
        self.assertIn("custom_llm", config.llm_blocks)
        self.assertNotIn("cheap_summarizer", config.llm_blocks)

    def test_custom_templates_preserved(self):
        """Test that custom templates are preserved."""
        custom_templates = {"custom_template": "Custom template content"}

        config = SummaryFlowConfig(
            name="summary_flow", prompt_templates=custom_templates
        )

        # Custom templates should be preserved, defaults not added
        self.assertEqual(len(config.prompt_templates), 1)
        self.assertIn("custom_template", config.prompt_templates)
        self.assertNotIn("summarize_chunk_template", config.prompt_templates)

    def test_mixed_custom_and_default_initialization(self):
        """Test initialization with some custom configs."""
        custom_llm_blocks = {"custom_llm": LLMConfig(model="custom-model")}

        config = SummaryFlowConfig(
            name="summary_flow", llm_blocks=custom_llm_blocks, chunk_size=1500
        )

        # Should have custom LLM blocks, not defaults
        self.assertEqual(len(config.llm_blocks), 1)
        self.assertIn("custom_llm", config.llm_blocks)

        # Should have default templates since none provided
        self.assertIn("summarize_chunk_template", config.prompt_templates)
        self.assertIn("combine_summaries_template", config.prompt_templates)

        # Custom chunk size should be preserved
        self.assertEqual(config.chunk_size, 1500)

    def test_chunking_configuration_options(self):
        """Test various chunking configuration options."""
        # Test disabling chunking
        config = SummaryFlowConfig(name="summary_flow", use_chunking=False)
        self.assertEqual(config.use_chunking, False)

        # Test custom chunk strategy
        def custom_chunker(text):
            return text.split("\n\n")

        config = SummaryFlowConfig(
            name="summary_flow",
            chunk_strategy=ChunkingStrategy.BY_CUSTOM,
            chunk_custom_strategy=custom_chunker,
        )
        self.assertEqual(config.chunk_strategy, ChunkingStrategy.BY_CUSTOM)
        self.assertEqual(config.chunk_custom_strategy, custom_chunker)

    def test_unsupported_chunk_strategy_raises_error(self):
        """Test that unsupported chunk strategies raise NotImplementedError."""
        with self.assertRaises(NotImplementedError) as context:
            SummaryFlowConfig(
                name="summary_flow", chunk_strategy=ChunkingStrategy.BY_SECTION
            )

        self.assertIn("not implemented", str(context.exception))

    def test_chunking_strategy_enum_values(self):
        """Test ChunkingStrategy enum values."""
        self.assertEqual(ChunkingStrategy.BY_SIZE.value, "by_size")
        self.assertEqual(ChunkingStrategy.BY_SECTION.value, "by_section")
        self.assertEqual(ChunkingStrategy.BY_CUSTOM.value, "by_custom")


if __name__ == "__main__":
    unittest.main()
