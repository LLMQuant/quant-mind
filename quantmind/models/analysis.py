"""Analysis and research-specific data models for QuantMind."""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

class PaperTag(BaseModel):
    """Paper tag with confidence score."""
    tag: str = Field(..., description="Tag name/category")
    value: str = Field(..., description="Tag value")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class QuestionAnswer(BaseModel):
    """Question and answer pair."""
    question: str = Field(..., description="Generated question")
    answer: str = Field(..., description="Generated answer")
    difficulty: str = Field(default="medium", description="Question difficulty")
    difficulty_level: str = Field(
        default="medium",
        description="Question difficulty level (backward compatibility)",
    )
    category: str = Field(default="general", description="Question category")
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.8, description="Answer confidence"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class PaperAnalysis(BaseModel):
    """Comprehensive paper analysis results."""
    paper_id: str
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    # Tags
    primary_tags: List[PaperTag] = Field(default_factory=list)
    secondary_tags: List[PaperTag] = Field(default_factory=list)
    # Q&A
    questions_answers: List[QuestionAnswer] = Field(default_factory=list)
    # Summary
    key_insights: List[str] = Field(default_factory=list)
    methodology_summary: Optional[str] = None
    results_summary: Optional[str] = None
    # Metadata
    analysis_version: str = "1.0"
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    analysis_duration: Optional[float] = None  # seconds
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

class AnalysisConfig(BaseModel):
    """Configuration for paper analysis."""
    # Tag analysis
    enable_tag_analysis: bool = True
    tag_confidence_threshold: float = 0.7
    max_primary_tags: int = 5
    max_secondary_tags: int = 10
    # Q&A generation
    enable_qa_generation: bool = True
    num_questions: int = 5
    include_different_difficulties: bool = True
    focus_on_insights: bool = True
    # Visual extraction
    enable_visual_extraction: bool = True
    extract_framework_only: bool = False
    min_importance_score: float = 0.6
    # LLM settings
    llm_model: str = "gpt-4o"
    max_tokens: int = 4000
    temperature: float = 0.3
    # Processing
    parallel_processing: bool = True
    cache_results: bool = True 