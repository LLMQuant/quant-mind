"""Tests for LLM-based research analysis."""

import pytest
from unittest.mock import Mock, patch

from quantmind.models.paper import Paper
from quantmind.research import (
    PaperAnalyzer,
    AnalysisConfig,
    LLMTagAnalyzer,
    LLMQAGenerator,
)
from quantmind.research.models import PaperTag, QuestionAnswer


class TestLLMTagAnalyzer:
    """Test LLM tag analyzer."""

    def test_init(self):
        """Test initialization."""
        config = AnalysisConfig()
        analyzer = LLMTagAnalyzer(config)
        assert analyzer.config == config

    @patch("quantmind.research.tag_analyzer.CAMEL_AVAILABLE", False)
    def test_init_no_camel(self):
        """Test initialization without CAMEL."""
        config = AnalysisConfig()
        analyzer = LLMTagAnalyzer(config)
        assert analyzer.client is None

    def test_get_analysis_text(self):
        """Test text preparation."""
        paper = Paper(
            title="Test Paper",
            abstract="Test abstract",
            full_text="Test content",
        )

        config = AnalysisConfig()
        analyzer = LLMTagAnalyzer(config)
        text = analyzer._get_analysis_text(paper)

        assert "Title: Test Paper" in text
        assert "Abstract: Test abstract" in text
        assert "Content: Test content" in text

    def test_generate_tag_summary(self):
        """Test tag summary generation."""
        primary_tags = [
            PaperTag(tag="equity", value="market", confidence=0.8),
            PaperTag(tag="high_frequency", value="frequency", confidence=0.9),
        ]
        secondary_tags = [
            PaperTag(tag="lstm", value="algorithm", confidence=0.7)
        ]

        config = AnalysisConfig()
        analyzer = LLMTagAnalyzer(config)
        summary = analyzer.generate_tag_summary(primary_tags, secondary_tags)

        assert "equity" in summary
        assert "high_frequency" in summary
        assert "lstm" in summary


class TestLLMQAGenerator:
    """Test LLM Q&A generator."""

    def test_init(self):
        """Test initialization."""
        config = AnalysisConfig()
        generator = LLMQAGenerator(config)
        assert generator.config == config

    @patch("quantmind.research.qa_generator.CAMEL_AVAILABLE", False)
    def test_init_no_camel(self):
        """Test initialization without CAMEL."""
        config = AnalysisConfig()
        generator = LLMQAGenerator(config)
        assert generator.client is None

    def test_prepare_context(self):
        """Test context preparation."""
        paper = Paper(
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author 1", "Author 2"],
            source="arxiv",
        )

        primary_tags = [PaperTag(tag="equity", value="market", confidence=0.8)]
        secondary_tags = [
            PaperTag(tag="lstm", value="algorithm", confidence=0.7)
        ]

        config = AnalysisConfig()
        generator = LLMQAGenerator(config)
        context = generator._prepare_context(
            paper, primary_tags, secondary_tags
        )

        assert context["title"] == "Test Paper"
        assert context["abstract"] == "Test abstract"
        assert context["authors"] == ["Author 1", "Author 2"]
        assert context["primary_tags"] == ["equity"]
        assert context["secondary_tags"] == ["lstm"]

    def test_get_insight_level(self):
        """Test insight level mapping."""
        config = AnalysisConfig()
        generator = LLMQAGenerator(config)

        assert generator._get_insight_level("beginner") == "basic"
        assert generator._get_insight_level("intermediate") == "intermediate"
        assert generator._get_insight_level("advanced") == "deep"
        assert generator._get_insight_level("expert") == "expert"

    def test_generate_qa_summary(self):
        """Test Q&A summary generation."""
        qa_pairs = [
            QuestionAnswer(
                question="What is the main contribution?",
                answer="A novel framework",
                difficulty="intermediate",
                difficulty_level="intermediate",
                category="methodology",
                confidence=0.8,
            ),
            QuestionAnswer(
                question="How to implement this?",
                answer="Use the provided code",
                difficulty="beginner",
                difficulty_level="beginner",
                category="implementation",
                confidence=0.7,
            ),
        ]

        config = AnalysisConfig()
        generator = LLMQAGenerator(config)
        summary = generator.generate_qa_summary(qa_pairs)

        assert "methodology" in summary
        assert "implementation" in summary
        assert "intermediate" in summary
        assert "beginner" in summary
