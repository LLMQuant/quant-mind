"""Base flow-cfg + input types shared across all flows.

`BaseFlowCfg` is the data contract for everything a flow exposes to YAML / CLI
users. Each `<Name>FlowCfg` subclasses it and adds domain knobs; nothing here
encodes flow behaviour. `BaseInput` is the parent of every flow's input
discriminated-union member; subclasses set a `Literal` discriminator field.
"""

from agents import ModelSettings
from pydantic import BaseModel, ConfigDict


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
