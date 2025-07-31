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

    @field_validator("encoding_format")
    def validate_encoding_format(cls, v):
        """Validate encoding format."""
        if v not in ["float", "base64"]:
            raise ValueError("encoding_format must be 'float' or 'base64'")
        return v

    def get_litellm_params(self) -> Dict[str, Any]:
        """Get parameters formatted for LiteLLM embedding."""
        params = {
            "model": self.model,
            "encoding_format": self.encoding_format,
            "timeout": self.timeout,
        }

        # Add optional parameters if provided
        if self.user:
            params["user"] = self.user
        if self.dimensions:
            params["dimensions"] = self.dimensions
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
        if (
            model_lower.startswith("text-embedding-")
            or model_lower.startswith("openai/")
            or "ada" in model_lower
            or "3" in model_lower
        ):
            return "openai"

        # Azure models
        elif "azure" in model_lower:
            return "azure"

        # Cohere models
        elif model_lower.startswith("embed-") or model_lower.startswith(
            "cohere/"
        ):
            return "cohere"

        # Default to openai for unknown models
        else:
            return "unknown"

    def create_variant(self, **overrides) -> "EmbeddingConfig":
        """Create a variant of this config with parameter overrides."""
        current_dict = self.model_dump()
        current_dict.update(overrides)
        return EmbeddingConfig(**current_dict)
