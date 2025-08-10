"""LLM configuration for QuantMind."""

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """Configuration for LLMBlock."""

    # Model configuration
    model: str = Field(
        default="gpt-4o", description="LLM model name (LiteLLM format)"
    )
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Temperature for generation"
    )
    max_tokens: int = Field(
        default=4000, gt=0, description="Maximum tokens to generate"
    )
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )

    # API configuration
    api_key: Optional[str] = Field(
        default=None, description="API key for the LLM provider"
    )
    base_url: Optional[str] = Field(
        default=None, description="Custom base URL for API"
    )
    api_version: Optional[str] = Field(
        default=None, description="API version (for Azure)"
    )

    # Request configuration
    timeout: int = Field(
        default=60, gt=0, description="Request timeout in seconds"
    )
    retry_attempts: int = Field(
        default=3, ge=0, description="Number of retry attempts"
    )
    retry_delay: float = Field(
        default=1.0, ge=0, description="Delay between retries in seconds"
    )

    # Additional provider-specific parameters
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the provider",
    )

    # System configuration
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for the model"
    )
    custom_instructions: Optional[str] = Field(
        default=None, description="Custom instructions to append"
    )

    @field_validator("model")
    def validate_model(cls, v):
        """Validate model name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Model name must be a non-empty string")
        return v.strip()

    @field_validator("api_key")
    def validate_api_key(cls, v):
        """Validate API key."""
        if v is not None and not isinstance(v, str):
            raise ValueError("API key must be a string")
        return v

    def get_effective_api_key(self) -> Optional[str]:
        """Get the effective API key with fallback logic.

        Priority:
        1. Directly provided api_key
        2. Environment variable specified in api_key_env_var
        3. Smart inference based on model provider type

        Returns:
            The effective API key or None if not found
        """
        # Priority 1: Direct API key
        if self.api_key and "${" not in self.api_key:
            return self.api_key

        # Priority 2: Smart inference based on provider
        provider = self.get_provider_type()
        smart_env_vars = {
            "openai": ["OPENAI_API_KEY", "OPENAI_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "azure": ["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
        }

        if provider in smart_env_vars:
            for env_var in smart_env_vars[provider]:
                key = os.getenv(env_var)
                if key:
                    return key

        return None

    def get_litellm_params(self) -> Dict[str, Any]:
        """Get parameters formatted for LiteLLM."""
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.timeout,
        }

        # Add API key if provided
        effective_api_key = self.get_effective_api_key()
        if effective_api_key:
            params["api_key"] = effective_api_key

        # Add base URL if provided
        if self.base_url:
            params["base_url"] = self.base_url

        # Add API version if provided (for Azure)
        if self.api_version:
            params["api_version"] = self.api_version

        # Add extra parameters
        params.update(self.extra_params)

        return params

    def get_provider_type(self) -> str:
        """Extract provider type from model name."""
        if self.model.startswith("gpt-") or self.model.startswith("openai/"):
            return "openai"
        elif self.model.startswith("claude-") or self.model.startswith(
            "anthropic/"
        ):
            return "anthropic"
        elif self.model.startswith("gemini-") or self.model.startswith(
            "google/"
        ):
            return "google"
        elif "azure" in self.model.lower():
            return "azure"
        elif "ollama" in self.model.lower():
            return "ollama"
        elif "deepseek" in self.model.lower():
            return "deepseek"
        else:
            return "unknown"

    def create_variant(self, **overrides) -> "LLMConfig":
        """Create a variant of this config with parameter overrides."""
        current_dict = self.model_dump()
        current_dict.update(overrides)
        return LLMConfig(**current_dict)
