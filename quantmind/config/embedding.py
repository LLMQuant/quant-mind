"""Embedding configuration for QuantMind."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class EmbeddingConfig(BaseModel):
    """Configuration for EmbeddingBlock."""

    # Model configuration
    model: str = Field(
        default="text-embedding-ada-002", description="Embedding model name"
    )

    # Optional parameters
    user: Optional[str] = Field(
        default=None,
        description="A unique identifier representing your end-user",
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="The number of dimensions the resulting output embeddings should have. Only supported in OpenAI/Azure text-embedding-3 and later models",
    )
    encoding_format: str = Field(
        default="float",
        description="The format to return the embeddings in. Can be either 'float' or 'base64'",
    )
    timeout: int = Field(
        default=600,
        description="The maximum time, in seconds, to wait for the API to respond",
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="The number of retry attempts",
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0,
        description="The delay between retries in seconds",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The api endpoint you want to call the model with",
    )
    api_version: Optional[str] = Field(
        default=None,
        description="(Azure-specific) the api version for the call",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The API key to authenticate and authorize requests. If not provided, the default API key is used",
    )
    api_type: Optional[str] = Field(
        default=None, description="The type of API to use"
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

    def get_litellm_params(self) -> Dict[str, Any]:
        """Get parameters formatted for LiteLLM embedding."""
        params = {
            "model": self.model,
        }

        # Add optional parameters if provided
        if self.user:
            params["user"] = self.user
        if self.dimensions:
            params["dimensions"] = self.dimensions
        if self.encoding_format:
            params["encoding_format"] = self.encoding_format
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_version:
            params["api_version"] = self.api_version
        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_type:
            params["api_type"] = self.api_type

        return params

    def get_provider_type(self) -> str:
        """Extract provider type from model name."""
        model_lower = self.model.lower()

        # OpenAI models
        if model_lower in [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]:
            return "openai"

        # Azure models
        elif "azure" in model_lower:
            return "azure"

        # Cohere models
        elif "gemini" in model_lower:
            return "gemini"

        # Default to openai for unknown models
        else:
            return "unknown"

    def create_variant(self, **overrides) -> "EmbeddingConfig":
        """Create a variant of this config with parameter overrides."""
        current_dict = self.model_dump()
        current_dict.update(overrides)
        return EmbeddingConfig(**current_dict)
