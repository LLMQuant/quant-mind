"""Unit tests for simplified LLM tagger."""

import json
import unittest
from unittest.mock import Mock, patch

from quantmind.config import LLMTaggerConfig
from quantmind.models.paper import Paper
from quantmind.tagger.llm_tagger import LLMTagger


class TestLLMTagger(unittest.TestCase):
    """Test cases for simplified LLM tagger."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_paper = Paper(
            title="Deep Learning for Cryptocurrency Trading",
            abstract="This paper presents LSTM networks for Bitcoin price prediction using sentiment analysis.",
            authors=["John Doe", "Jane Smith"],
            url="https://example.com/paper.pdf",
        )

    def test_tagger_initialization(self):
        """Test tagger initialization with default parameters."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            tagger = LLMTagger()

            self.assertEqual(tagger.llm_type, "openai")
            self.assertEqual(tagger.llm_name, "gpt-4o")
            self.assertEqual(tagger.config.max_tags, 5)
            self.assertEqual(tagger.config.llm_config.temperature, 0.0)

    def test_tagger_initialization_with_params(self):
        """Test tagger initialization with custom parameters."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            tagger = LLMTagger(
                config=LLMTaggerConfig.create(
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    max_tags=3,
                    custom_instructions="Custom instructions for tagging",
                )
            )

            self.assertEqual(tagger.llm_name, "gpt-3.5-turbo")
            self.assertEqual(tagger.config.max_tags, 3)
            self.assertEqual(tagger.config.llm_config.temperature, 0.7)
            self.assertEqual(
                tagger.config.llm_config.custom_instructions,
                "Custom instructions for tagging",
            )

    def test_tagger_initialization_with_direct_config(self):
        """Test tagger initialization with direct config creation."""
        from quantmind.config.llm import LLMConfig

        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            llm_config = LLMConfig(
                model="claude-3-5-sonnet-20241022",
                temperature=0.5,
                max_tokens=3000,
                api_key="test-key",
            )

            tagger_config = LLMTaggerConfig(
                llm_config=llm_config,
                max_tags=7,
                custom_prompt="Analyze content: {content} and return {max_tags} tags",
            )

            tagger = LLMTagger(config=tagger_config)

            self.assertEqual(tagger.llm_name, "claude-3-5-sonnet-20241022")
            self.assertEqual(tagger.config.max_tags, 7)
            self.assertEqual(tagger.config.llm_config.temperature, 0.5)
            self.assertEqual(tagger.config.llm_config.max_tokens, 3000)
            self.assertEqual(tagger.config.llm_config.api_key, "test-key")

    def test_prepare_content(self):
        """Test content preparation from paper."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            tagger = LLMTagger()
            content = tagger._prepare_content(self.sample_paper)

            self.assertIn(
                "Title: Deep Learning for Cryptocurrency Trading", content
            )
            self.assertIn("Abstract: This paper presents LSTM", content)

    def test_build_default_prompt(self):
        """Test default prompt construction."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            tagger = LLMTagger()
            content = "Test content"

            prompt = tagger._build_prompt(content)

            self.assertIn("quantitative finance", prompt)
            self.assertIn("Test content", prompt)
            self.assertIn("5 relevant tags", prompt)
            self.assertIn("JSON list", prompt)

    def test_build_prompt_with_custom_prompt(self):
        """Test prompt construction with custom prompt."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            tagger = LLMTagger(
                config=LLMTaggerConfig(
                    custom_prompt="Analyze the content and return 5 relevant tags: {content}"
                )
            )
            content = "Test content"

            prompt = tagger._build_prompt(content)

            self.assertIn(
                "Analyze the content and return 5 relevant tags", prompt
            )
            self.assertIn("Test content", prompt)

    def test_build_custom_prompt_with_variables(self):
        """Test custom prompt construction with variables."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            custom_prompt = "Analyze: {content} and return {max_tags} tags"
            tagger = LLMTagger(
                config=LLMTaggerConfig(custom_prompt=custom_prompt)
            )
            content = "Test content"

            prompt = tagger._build_prompt(content)

            self.assertEqual(prompt, "Analyze: Test content and return 5 tags")

    def test_parse_tags_json(self):
        """Test parsing tags from JSON response."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            tagger = LLMTagger()
            response = (
                '["crypto", "machine learning", "lstm", "bitcoin", "trading"]'
            )

            tags = tagger._parse_tags(response)

            self.assertEqual(len(tags), 5)
            self.assertIn("crypto", tags)
            self.assertIn("machine learning", tags)

    def test_parse_tags_json_with_extra_text(self):
        """Test parsing tags from JSON response with extra text."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            tagger = LLMTagger()
            response = 'Here are the tags: ["crypto", "deep learning", "sentiment"] for this paper.'

            tags = tagger._parse_tags(response)

            self.assertEqual(len(tags), 3)
            self.assertIn("crypto", tags)
            self.assertIn("deep learning", tags)

    def test_parse_tags_fallback(self):
        """Test fallback tag parsing from plain text."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            tagger = LLMTagger()
            response = (
                '"crypto", "machine learning", "trading", "sentiment analysis"'
            )

            tags = tagger._parse_tags(response)

            self.assertTrue(len(tags) > 0)
            self.assertIn("crypto", tags)

    def test_tag_paper_success(self):
        """Test successful paper tagging."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            # Mock LLMBlock
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            # Mock LLM response
            mock_llm_block.generate_text.return_value = (
                '["crypto", "lstm", "trading", "deep learning", "bitcoin"]'
            )

            tagger = LLMTagger()
            result_paper = tagger.tag_paper(self.sample_paper)

            # Check that tags were added
            self.assertTrue(len(result_paper.tags) > 0)
            self.assertIn("crypto", result_paper.tags)
            self.assertIn("llm_tagger", result_paper.meta_info["tagger"])

    def test_tag_paper_no_llm_block(self):
        """Test paper tagging when no LLM block is available."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_create.return_value = None

            tagger = LLMTagger()
            tagger.llm_block = None

            result_paper = tagger.tag_paper(self.sample_paper)

            # Paper should be returned unchanged
            self.assertEqual(result_paper.title, self.sample_paper.title)
            self.assertEqual(len(result_paper.tags), 0)

    def test_extract_tags(self):
        """Test tag extraction from arbitrary text."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            # Mock LLMBlock
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            # Mock LLM response
            mock_llm_block.generate_text.return_value = (
                '["finance", "analysis", "data"]'
            )

            # Configure tagger to expect 3 tags
            config = LLMTaggerConfig.create(max_tags=3)
            tagger = LLMTagger(config=config)

            tags = tagger.extract_tags(
                "Financial data analysis paper", "Finance Title"
            )

            self.assertEqual(len(tags), 3)
            self.assertIn("finance", tags)

    def test_extract_tags_from_text_quoted(self):
        """Test extracting tags from text with quoted items."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            tagger = LLMTagger()
            text = 'The tags are "machine learning", "trading", "analysis"'

            tags = tagger._extract_tags_from_text(text)

            self.assertEqual(len(tags), 3)
            self.assertIn("machine learning", tags)
            self.assertIn("trading", tags)

    def test_extract_tags_from_text_comma_separated(self):
        """Test extracting tags from comma-separated text."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            tagger = LLMTagger()
            text = "machine learning, deep learning, trading algorithms, risk management"

            tags = tagger._extract_tags_from_text(text)

            self.assertTrue(len(tags) >= 3)
            self.assertIn("machine learning", tags)

    def test_max_tags_limit(self):
        """Test that tag count is limited to max_tags."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            tagger = LLMTagger(config=LLMTaggerConfig(max_tags=3))
            response = '["tag1", "tag2", "tag3", "tag4", "tag5"]'

            tags = tagger._parse_tags(response)
            limited_tags = tags[: tagger.config.max_tags]

            self.assertEqual(len(limited_tags), 3)

    def test_llm_config_access(self):
        """Test accessing LLM configuration through composition."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            config = LLMTaggerConfig.create(
                model="gpt-4o-mini",
                temperature=0.8,
                max_tokens=2000,
                api_key="test-api-key",
                max_tags=8,
            )

            tagger = LLMTagger(config=config)

            # Test access to LLM config properties
            self.assertEqual(tagger.config.llm_config.model, "gpt-4o-mini")
            self.assertEqual(tagger.config.llm_config.temperature, 0.8)
            self.assertEqual(tagger.config.llm_config.max_tokens, 2000)
            self.assertEqual(tagger.config.llm_config.api_key, "test-api-key")
            self.assertEqual(tagger.config.max_tags, 8)

    def test_provider_detection(self):
        """Test LLM provider type detection."""
        with patch(
            "quantmind.tagger.llm_tagger.create_llm_block"
        ) as mock_create:
            mock_llm_block = Mock()
            mock_create.return_value = mock_llm_block

            # Test OpenAI
            config = LLMTaggerConfig.create(model="gpt-4o")
            tagger = LLMTagger(config=config)
            self.assertEqual(
                tagger.config.llm_config.get_provider_type(), "openai"
            )

            # Test Anthropic
            config = LLMTaggerConfig.create(model="claude-3-5-sonnet-20241022")
            tagger = LLMTagger(config=config)
            self.assertEqual(
                tagger.config.llm_config.get_provider_type(), "anthropic"
            )

            # Test Google
            config = LLMTaggerConfig.create(model="gemini-1.5-pro")
            tagger = LLMTagger(config=config)
            self.assertEqual(
                tagger.config.llm_config.get_provider_type(), "google"
            )


if __name__ == "__main__":
    unittest.main()
