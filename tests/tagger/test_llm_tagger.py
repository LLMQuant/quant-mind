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
        tagger = LLMTagger()

        self.assertEqual(tagger.llm_type, "openai")
        self.assertEqual(tagger.llm_name, "gpt-4o")
        self.assertEqual(tagger.config.max_tags, 5)
        self.assertEqual(tagger.config.temperature, 0.3)

    def test_tagger_initialization_with_params(self):
        """Test tagger initialization with custom parameters."""
        tagger = LLMTagger(
            config=LLMTaggerConfig(
                llm_name="gpt-3.5-turbo",
                max_tags=3,
                temperature=0.7,
                custom_prompt="Custom prompt: {content}",
            )
        )

        self.assertEqual(tagger.llm_name, "gpt-3.5-turbo")
        self.assertEqual(tagger.config.max_tags, 3)
        self.assertEqual(tagger.config.temperature, 0.7)
        self.assertEqual(
            tagger.config.custom_prompt, "Custom prompt: {content}"
        )

    def test_prepare_content(self):
        """Test content preparation from paper."""
        tagger = LLMTagger()

        content = tagger._prepare_content(self.sample_paper)

        self.assertIn(
            "Title: Deep Learning for Cryptocurrency Trading", content
        )
        self.assertIn("Abstract: This paper presents LSTM", content)

    def test_build_default_prompt(self):
        """Test default prompt construction."""
        tagger = LLMTagger()
        content = "Test content"

        prompt = tagger._build_prompt(content)

        self.assertIn("quantitative finance", prompt)
        self.assertIn("Test content", prompt)
        self.assertIn("5 relevant tags", prompt)
        self.assertIn("JSON list", prompt)

    def test_build_prompt_with_instructions(self):
        """Test prompt construction with instructions."""
        tagger = LLMTagger(
            config=LLMTaggerConfig(
                custom_instructions="Analyze the content and return 5 relevant tags"
            )
        )
        content = "Test content"

        prompt = tagger._build_prompt(content)

        self.assertIn("Analyze the content and return 5 relevant tags", prompt)

    def test_build_custom_prompt(self):
        """Test custom prompt construction."""
        custom_prompt = "Analyze: {content} and return {max_tags} tags"
        tagger = LLMTagger(config=LLMTaggerConfig(custom_prompt=custom_prompt))
        content = "Test content"

        prompt = tagger._build_prompt(content)

        self.assertEqual(prompt, "Analyze: Test content and return 5 tags")

    def test_parse_tags_json(self):
        """Test parsing tags from JSON response."""
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
        tagger = LLMTagger()
        response = 'Here are the tags: ["crypto", "deep learning", "sentiment"] for this paper.'

        tags = tagger._parse_tags(response)

        self.assertEqual(len(tags), 3)
        self.assertIn("crypto", tags)
        self.assertIn("deep learning", tags)

    def test_parse_tags_fallback(self):
        """Test fallback tag parsing from plain text."""
        tagger = LLMTagger()
        response = (
            '"crypto", "machine learning", "trading", "sentiment analysis"'
        )

        tags = tagger._parse_tags(response)

        self.assertTrue(len(tags) > 0)
        self.assertIn("crypto", tags)

    @patch("openai.OpenAI")
    def test_tag_paper_success(self, mock_openai):
        """Test successful paper tagging."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = (
            '["crypto", "lstm", "trading", "deep learning", "bitcoin"]'
        )
        mock_client.chat.completions.create.return_value = mock_response

        tagger = LLMTagger()
        tagger.client = mock_client

        result_paper = tagger.tag_paper(self.sample_paper)

        # Check that tags were added
        self.assertTrue(len(result_paper.tags) > 0)
        self.assertIn("crypto", result_paper.tags)
        self.assertIn("llm_tagger", result_paper.meta_info["tagger"])

    def test_tag_paper_no_client(self):
        """Test paper tagging when no LLM client is available."""
        tagger = LLMTagger()
        tagger.client = None

        result_paper = tagger.tag_paper(self.sample_paper)

        # Paper should be returned unchanged
        self.assertEqual(result_paper.title, self.sample_paper.title)
        self.assertEqual(len(result_paper.tags), 0)

    @patch("openai.OpenAI")
    def test_extract_tags(self, mock_openai):
        """Test tag extraction from arbitrary text."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = '["finance", "analysis", "data"]'
        mock_client.chat.completions.create.return_value = mock_response

        tagger = LLMTagger()
        tagger.client = mock_client

        tags = tagger.extract_tags(
            "Financial data analysis paper", "Finance Title"
        )

        self.assertEqual(len(tags), 3)
        self.assertIn("finance", tags)

    def test_extract_tags_from_text_quoted(self):
        """Test extracting tags from text with quoted items."""
        tagger = LLMTagger()
        text = 'The tags are "machine learning", "trading", "analysis"'

        tags = tagger._extract_tags_from_text(text)

        self.assertEqual(len(tags), 3)
        self.assertIn("machine learning", tags)
        self.assertIn("trading", tags)

    def test_extract_tags_from_text_comma_separated(self):
        """Test extracting tags from comma-separated text."""
        tagger = LLMTagger()
        text = "machine learning, deep learning, trading algorithms, risk management"

        tags = tagger._extract_tags_from_text(text)

        self.assertTrue(len(tags) >= 3)
        self.assertIn("machine learning", tags)

    def test_max_tags_limit(self):
        """Test that tag count is limited to max_tags."""
        tagger = LLMTagger(config=LLMTaggerConfig(max_tags=3))
        response = '["tag1", "tag2", "tag3", "tag4", "tag5"]'

        tags = tagger._parse_tags(response)
        limited_tags = tags[: tagger.config.max_tags]

        self.assertEqual(len(limited_tags), 3)


if __name__ == "__main__":
    unittest.main()
