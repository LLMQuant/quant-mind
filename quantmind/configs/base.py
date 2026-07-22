"""Base flow-cfg + input types shared across all flows.

`BaseFlowCfg` is the data contract for everything a flow exposes to YAML / CLI
users. Each `<Name>FlowCfg` subclasses it and adds domain knobs; nothing here
encodes flow behaviour. `BaseInput` is the parent of every flow's input
discriminated-union member; subclasses set a `Literal` discriminator field.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agents import ModelSettings
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from agents.models.interface import Model

ATLASCLOUD_BASE_URL = "https://api.atlascloud.ai/v1"
ATLASCLOUD_DEFAULT_CHAT_MODEL = "qwen/qwen3.5-flash"
ATLASCLOUD_DEFAULT_REASONING_MODEL = "deepseek-ai/deepseek-v4-pro"
ATLASCLOUD_MODEL_PREFIXES = ("atlascloud/", "atlas-cloud/", "atlas/")


class BaseFlowCfg(BaseModel):
    """Base configuration shared by all flows."""

    model_config = ConfigDict(extra="forbid")

    # Model & execution
    model: str = "gpt-4o"
    model_settings: ModelSettings | None = None
    max_turns: int = 10
    timeout_seconds: float = 300.0

    # Output persistence
    output_dir: str | None = None
    overwrite: bool = False

    # Mind / memory (filesystem-backed when set)
    memory_dir: str | None = None

    # Observability (consumed by flows/_runner in PR5)
    workflow_name: str | None = None
    trace_metadata: dict[str, str] | None = None
    trace_include_sensitive_data: bool = True
    tracing_disabled: bool = False
    archive_trajectory: bool = True

    # Cost / budget guardrails (enforced in PR5+)
    max_total_input_tokens: int | None = None
    max_total_cost_usd: float | None = None

    # Default guardrails (populated in PR8+)
    enable_default_guardrails: bool = True


class BaseInput(BaseModel):
    """Parent of every flow's discriminated-union input member."""

    model_config = ConfigDict(extra="forbid")


def atlascloud_model(model: str = ATLASCLOUD_DEFAULT_CHAT_MODEL) -> str:
    """Return a QuantMind model value routed through Atlas Cloud.

    The returned value can be used anywhere ``BaseFlowCfg.model`` is accepted.
    Runtime credentials are read from ``ATLASCLOUD_API_KEY`` (or
    ``ATLAS_CLOUD_API_KEY``) when the flow actually builds its SDK agent.
    """
    value = model.strip()
    if not value:
        raise ValueError("atlascloud model name must not be blank")
    if is_atlascloud_model(value):
        return value
    return f"atlascloud/{value}"


def is_atlascloud_model(model: str) -> bool:
    """Whether a model string uses one of QuantMind's Atlas Cloud aliases."""
    return model.startswith(ATLASCLOUD_MODEL_PREFIXES)


def resolve_agent_model(model: str) -> str | Model:
    """Resolve a config model string into an Agents SDK model.

    Non-Atlas values are returned unchanged. ``atlascloud/...`` aliases are
    backed by the Agents SDK LiteLLM adapter with Atlas Cloud's
    OpenAI-compatible endpoint and API key environment variables.
    """
    if not is_atlascloud_model(model):
        return model

    from agents.extensions.models.litellm_model import LitellmModel

    return LitellmModel(
        model=f"openai/{_strip_atlascloud_prefix(model)}",
        base_url=_atlascloud_base_url(),
        api_key=_atlascloud_api_key(),
    )


def _strip_atlascloud_prefix(model: str) -> str:
    for prefix in ATLASCLOUD_MODEL_PREFIXES:
        if model.startswith(prefix):
            value = model[len(prefix) :].strip()
            if not value:
                raise ValueError("atlascloud model name must not be blank")
            return value
    raise ValueError("model is not an Atlas Cloud alias")


def _atlascloud_api_key() -> str:
    value = os.getenv("ATLASCLOUD_API_KEY") or os.getenv("ATLAS_CLOUD_API_KEY")
    if not value:
        raise ValueError(
            "Set ATLASCLOUD_API_KEY before using atlascloud/... models"
        )
    return value


def _atlascloud_base_url() -> str:
    return (
        os.getenv("ATLASCLOUD_API_BASE")
        or os.getenv("ATLASCLOUD_BASE_URL")
        or os.getenv("ATLAS_CLOUD_API_BASE")
        or os.getenv("ATLAS_CLOUD_BASE_URL")
        or ATLASCLOUD_BASE_URL
    ).rstrip("/")
