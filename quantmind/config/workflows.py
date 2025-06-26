"""Workflow-specific configuration models for QuantMind."""

from typing import List, Optional

from pydantic import BaseModel, Field


class BaseWorkflowConfig(BaseModel):
    """Base configuration for all workflows."""

    # LLM settings
    llm_type: str = Field(default="openai", description="LLM provider type")
    llm_name: str = Field(default="gpt-4o", description="LLM model name")
    api_key: Optional[str] = Field(default=None, description="API key for LLM")
    base_url: Optional[str] = Field(
        default=None, description="Custom API base URL"
    )

    # Generation settings
    temperature: float = Field(
        default=0.3, ge=0.0, le=2.0, description="LLM temperature"
    )
    max_tokens: int = Field(default=4000, gt=0, description="Maximum tokens")

    # Processing settings
    timeout: int = Field(
        default=300, gt=0, description="Request timeout in seconds"
    )
    retry_attempts: int = Field(
        default=3, ge=0, description="Number of retry attempts"
    )

    # Custom instructions
    custom_instructions: Optional[str] = Field(
        default=None, description="Custom instructions to append to prompts"
    )


class QAWorkflowConfig(BaseWorkflowConfig):
    """Configuration for Q&A generation workflow."""

    # Q&A generation settings
    num_questions: int = Field(
        default=5, ge=1, description="Number of questions to generate"
    )
    include_different_difficulties: bool = Field(
        default=True,
        description="Generate questions with different difficulty levels",
    )
    focus_on_insights: bool = Field(
        default=True, description="Focus on generating insightful questions"
    )

    # Question categories
    question_categories: List[str] = Field(
        default_factory=lambda: [
            "basic_understanding",
            "methodology",
            "technical_details",
            "critical_analysis",
            "future_directions",
        ],
        description="Categories of questions to generate",
    )

    # Difficulty levels
    difficulty_levels: List[str] = Field(
        default_factory=lambda: [
            "beginner",
            "intermediate",
            "advanced",
            "expert",
        ],
        description="Difficulty levels for questions",
    )


class AnalyzerWorkflowConfig(BaseWorkflowConfig):
    """Configuration for paper analysis workflow."""

    # Tag analysis settings
    enable_tag_analysis: bool = Field(
        default=True, description="Enable tag analysis"
    )
    max_tags: int = Field(
        default=10, ge=1, description="Maximum number of tags to generate"
    )
    tag_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for tags",
    )

    # Analysis categories
    primary_categories: List[str] = Field(
        default_factory=lambda: [
            "market_type",
            "frequency",
            "methodology",
            "application",
        ],
        description="Primary tag categories",
    )

    secondary_categories: List[str] = Field(
        default_factory=lambda: [
            "data_source",
            "algorithm",
            "performance_metric",
            "risk_measure",
        ],
        description="Secondary tag categories",
    )

    # Summary generation
    generate_methodology_summary: bool = Field(
        default=True, description="Generate methodology summary"
    )
    generate_results_summary: bool = Field(
        default=True, description="Generate results summary"
    )
    summary_max_length: int = Field(
        default=500, ge=100, description="Maximum length for summaries"
    )
