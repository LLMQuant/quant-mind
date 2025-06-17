"""Tests for RuleTagger."""

import pytest
from quantmind.models.paper import Paper
from quantmind.tagger.rule_tagger import RuleTagger


class TestRuleTagger:
    """Test cases for the RuleTagger."""

    def test_tagger_initialization(self):
        """Test tagger initialization."""
        tagger = RuleTagger()

        assert tagger.name == "rule"
        assert not tagger.case_sensitive
        assert "Machine Learning in Finance" in tagger.category_rules

    def test_custom_config(self):
        """Test tagger with custom configuration."""
        config = {
            "case_sensitive": True,
            "category_rules": {"Custom Category": ["custom", "keyword"]},
        }

        tagger = RuleTagger(config=config)

        assert tagger.case_sensitive
        assert "Custom Category" in tagger.category_rules

    def test_extract_categories(self):
        """Test category extraction."""
        tagger = RuleTagger()

        text = "This paper discusses machine learning applications in financial markets"
        categories = tagger.extract_categories(text)

        # RuleTagger returns normalized (lowercase) categories
        assert "Machine Learning in Finance" in categories

    def test_extract_categories_case_insensitive(self):
        """Test case insensitive category extraction."""
        tagger = RuleTagger(config={"case_sensitive": False})

        text = "This paper discusses MACHINE LEARNING applications"
        categories = tagger.extract_categories(text)

        assert "Machine Learning in Finance" in categories

    def test_extract_categories_case_sensitive(self):
        """Test case sensitive category extraction."""
        tagger = RuleTagger(config={"case_sensitive": True})

        text = "This paper discusses MACHINE LEARNING applications"
        categories = tagger.extract_categories(text)

        # Should not match due to case sensitivity
        assert "Machine Learning in Finance" not in categories

    def test_extract_tags(self):
        """Test tag extraction."""
        tagger = RuleTagger()

        text = "This paper presents a new algorithm to predict financial markets using neural networks"
        tags = tagger.extract_tags(text)

        assert "algorithm" in tags
        assert "finance" in tags
        assert "prediction" in tags  # Should match 'predict' in the text

    def test_tag_paper(self):
        """Test paper tagging."""
        tagger = RuleTagger()

        paper = Paper(
            title="Deep Learning for Portfolio Optimization",
            abstract="This paper presents deep learning techniques for portfolio optimization in financial markets",
        )

        tagged_paper = tagger.tag_paper(paper)

        assert "Deep Learning in Finance" in tagged_paper.categories
        assert "Portfolio Optimization" in tagged_paper.categories
        assert len(tagged_paper.tags) > 0
        assert tagged_paper.meta_info["tagger"] == "rule"

    def test_tag_papers_batch(self):
        """Test batch paper tagging."""
        tagger = RuleTagger()

        papers = [
            Paper(
                title="Machine Learning in Trading",
                abstract="ML applications in algorithmic trading",
            ),
            Paper(
                title="Time Series Forecasting",
                abstract="Predicting financial time series using statistical methods",
            ),
        ]

        tagged_papers = tagger.tag_papers(papers)

        assert len(tagged_papers) == 2
        assert "Machine Learning in Finance" in tagged_papers[0].categories
        assert "Time Series Forecasting" in tagged_papers[1].categories

    def test_confidence_score(self):
        """Test confidence score calculation."""
        tagger = RuleTagger()

        paper1 = Paper(title="Test", abstract="Test")  # No matches
        paper2 = Paper(
            title="ML Finance", abstract="machine learning trading"
        )  # Some matches

        # Tag papers
        tagged1 = tagger.tag_paper(paper1)
        tagged2 = tagger.tag_paper(paper2)

        score1 = tagger.get_confidence_score(tagged1)
        score2 = tagger.get_confidence_score(tagged2)

        assert score1 < score2  # More matches should have higher confidence
        assert 0.0 <= score1 <= 1.0
        assert 0.0 <= score2 <= 1.0

    def test_add_category_rule(self):
        """Test adding custom category rules."""
        tagger = RuleTagger()

        # Add custom rule
        tagger.add_category_rule(
            "Custom Finance", ["cryptocurrency", "blockchain"]
        )

        text = "This paper discusses cryptocurrency trading strategies"
        categories = tagger.extract_categories(text)

        assert "Custom Finance" in categories

    def test_add_tag_pattern(self):
        """Test adding custom tag patterns."""
        tagger = RuleTagger()

        # Add custom pattern
        tagger.add_tag_pattern(
            "crypto", r"\b(bitcoin|ethereum|cryptocurrency)\b"
        )

        text = "This paper analyzes bitcoin price movements"
        tags = tagger.extract_tags(text)

        assert "crypto" in tags

    def test_financial_terms_extraction(self):
        """Test extraction of specific financial terms."""
        tagger = RuleTagger()

        text = "We analyze stock and bond markets using regression analysis"
        tags = tagger.extract_tags(text)

        assert "stock" in tags
        assert "bond" in tags
        assert "regression" in tags

    def test_category_statistics(self):
        """Test category statistics."""
        tagger = RuleTagger()

        papers = [
            Paper(
                title="ML Paper", abstract="machine learning", categories=["ML"]
            ),
            Paper(
                title="DL Paper",
                abstract="deep learning",
                categories=["ML", "DL"],
            ),
            Paper(
                title="Finance Paper",
                abstract="trading",
                categories=["Finance"],
            ),
        ]

        stats = tagger.get_category_stats(papers)

        assert stats["ML"] == 2
        assert stats["DL"] == 1
        assert stats["Finance"] == 1

    def test_validate_tags(self):
        """Test tag validation."""
        tagger = RuleTagger()

        raw_tags = [
            "  valid_tag  ",
            "",
            "another_tag",
            None,
            "duplicate",
            "duplicate",
        ]
        validated = tagger.validate_tags(raw_tags)

        assert "valid_tag" in validated
        assert "another_tag" in validated
        assert (
            len([t for t in validated if t == "duplicate"]) == 1
        )  # No duplicates
        assert "" not in validated  # No empty strings

    def test_validate_categories(self):
        """Test category validation."""
        tagger = RuleTagger()

        raw_categories = [
            "Valid Category",
            "  Another Category  ",
            "",
            "Valid Category",
        ]
        validated = tagger.validate_categories(raw_categories)

        assert "Valid Category" in validated  # Categories preserve case
        assert "Another Category" in validated
        assert (
            len([c for c in validated if c == "Valid Category"]) == 1
        )  # No duplicates
