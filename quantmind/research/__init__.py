"""QuantMind Research Analysis Module.

This module provides advanced analysis capabilities for research papers including:
- LLM-based paper tagging and categorization
- LLM-based deep Q&A generation with insights
- Comprehensive analysis summary
"""

from quantmind.research.paper_analyzer import PaperAnalyzer
from quantmind.research.tag_analyzer import LLMTagAnalyzer
from quantmind.research.qa_generator import LLMQAGenerator
from quantmind.research.models import AnalysisConfig, PaperTag, QuestionAnswer

__all__ = [
    "LLMTagAnalyzer",
    "LLMQAGenerator",
    "PaperAnalysis",
    "AnalysisConfig",
    "PaperTag",
    "QuestionAnswer",
]
