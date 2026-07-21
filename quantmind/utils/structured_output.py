"""Structured-output helpers for JSON-object-only model routes."""

import json
from dataclasses import replace
from typing import Any, TypeVar, cast

from agents import ModelSettings
from pydantic import BaseModel

_OutputT = TypeVar("_OutputT", bound=BaseModel)


def requires_json_object_mode(model: str) -> bool:
    """Return whether a model is routed through LiteLLM.

    LiteLLM can route providers that implement only OpenAI-compatible JSON mode.
    JSON-object mode works across those providers, while strict ``json_schema``
    is not universally available. Native OpenAI routes keep the SDK's stricter
    output contract.
    """
    return model.lower().startswith("litellm/")


def json_object_instructions(
    instructions: str,
    output_type: type[_OutputT],
) -> str:
    """Append a local-validation contract and schema to agent instructions."""
    schema = json.dumps(output_type.model_json_schema(), ensure_ascii=False)
    return (
        f"{instructions}\n\n"
        "IMPORTANT — final output format:\n"
        "After completing any required tool calls, return ONLY one JSON object "
        "with no prose or Markdown fences. It must conform to this JSON Schema; "
        "use canonical UUID strings where the schema requires UUID values.\n\n"
        f"```json\n{schema}\n```"
    )


def json_object_model_settings(
    settings: ModelSettings | None,
) -> ModelSettings:
    """Force JSON-object mode while preserving other model settings."""
    base = settings or ModelSettings()
    extra_body: dict[str, Any] = dict(
        cast(dict[str, Any], base.extra_body or {})
    )
    extra_body["response_format"] = {"type": "json_object"}
    return replace(base, extra_body=cast(Any, extra_body))


def validate_json_object(
    value: object,
    output_type: type[_OutputT],
) -> _OutputT:
    """Validate a JSON-mode final output against its Pydantic output type."""
    if isinstance(value, str):
        return output_type.model_validate_json(_strip_json_fence(value))
    return output_type.model_validate(value)


def _strip_json_fence(value: str) -> str:
    """Strip one optional Markdown code fence around a JSON response."""
    content = value.strip()
    if not content.startswith("```"):
        return content
    lines = content.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()
