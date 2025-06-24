"""Configuration models for taggers."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class BaseTaggerConfig(BaseModel):
    """Base configuration for all taggers."""

    max_tags: int = Field(default=5, ge=1, le=10)
    meta_info: Optional[Dict[str, Any]] = Field(default=None)


class LLMTaggerConfig(BaseTaggerConfig):
    """Configuration for LLM-based tagger."""

    llm_type: str = Field(default="openai")
    llm_name: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    custom_instructions: Optional[str] = Field(default=None)
    custom_prompt: Optional[str] = Field(default=None)
    max_tokens: int = Field(default=5_000, ge=100, le=1_000_000)

    @field_validator("llm_type")
    def validate_llm_type(cls, v: str) -> str:
        """Validate model type."""
        if v not in ["openai", "anthropic", "gemini", "deepseek"]:
            raise ValueError("llm_type must be one of openai or anthropic")
        return v
