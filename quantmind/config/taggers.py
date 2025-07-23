"""Configuration models for taggers."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from quantmind.config.llm import LLMConfig


class BaseTaggerConfig(BaseModel):
    """Base configuration for all taggers."""

    max_tags: int = Field(default=5, ge=1, le=10)
    meta_info: Optional[Dict[str, Any]] = Field(default=None)


class LLMTaggerConfig(BaseTaggerConfig):
    """Configuration for LLM-based tagger using LLMConfig composition."""

    # LLM configuration - using composition pattern to avoid field duplication
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig, description="LLM configuration"
    )

    # Tagger-specific settings
    custom_prompt: Optional[str] = Field(
        default=None, description="Custom tagging prompt"
    )

    @classmethod
    def create(
        cls,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 5000,
        max_tags: int = 5,
        custom_instructions: Optional[str] = None,
        **kwargs,
    ) -> "LLMTaggerConfig":
        """Create an LLMTaggerConfig with convenient LLM parameter specification.

        Args:
            model: LLM model name
            api_key: API key for LLM
            temperature: LLM temperature
            max_tokens: Maximum tokens
            max_tags: Maximum number of tags to generate
            custom_instructions: Custom instructions to append to prompts
            **kwargs: Additional tagger-specific parameters

        Returns:
            Configured LLMTaggerConfig instance
        """
        llm_config = LLMConfig(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_instructions=custom_instructions,
        )
        return cls(llm_config=llm_config, max_tags=max_tags, **kwargs)
