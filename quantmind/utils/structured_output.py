"""Structured-output helper with a transparent JSON-object fallback.

``run_structured`` runs an agent for a Pydantic output type and hides one
cross-provider wrinkle: setting ``output_type=`` makes the Agents SDK send a
strict ``response_format={"type": "json_schema"}``, which some LiteLLM-routed
providers (DeepSeek, ...) reject outright at request time. When that happens the
helper re-runs the same call once in JSON-object mode — no ``output_type``, the
schema pinned into the instructions, ``response_format={"type": "json_object"}``
— and validates the raw output locally. Callers pass only ``cfg.model``; the
fallback is invisible to them.

The helper is layer-agnostic: each call site supplies ``build_agent`` (how the
agent is constructed) and ``run`` (which runner executes it — ``flows`` uses
``run_with_observability``, ``mind`` uses ``Runner.run``). That keeps the shared
logic in ``utils`` (a leaf both layers import) without ``mind`` importing
``flows``.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, TypeVar, cast

from agents import Agent, ModelSettings
from openai import BadRequestError
from pydantic import BaseModel

_OutputT = TypeVar("_OutputT", bound=BaseModel)

# Substrings marking a provider that rejects the strict ``json_schema``
# response_format, so we fall back rather than swallow an unrelated bad request.
_RESPONSE_FORMAT_MARKERS = (
    "response_format",
    "response format",
    "json_schema",
    "json schema",
    "structured output",
)


async def run_structured(
    output_type: type[_OutputT],
    *,
    build_agent: Callable[[bool], Agent[Any]],
    run: Callable[[Agent[Any]], Awaitable[Any]],
) -> _OutputT:
    """Run an agent for ``output_type``, falling back to JSON-object mode.

    Args:
        output_type: The Pydantic type the final output must validate against.
        build_agent: Builds the agent for a mode. Called with ``False`` for the
            native strict ``json_schema`` path (agent carries ``output_type=``)
            and, only if the provider rejects that format, with ``True`` for the
            JSON-object fallback (no ``output_type``; schema pinned into the
            instructions and ``response_format`` forced to ``json_object`` via
            ``json_object_instructions`` / ``json_object_model_settings``).
        run: Executes one agent and returns its final output — a validated model
            on the strict path, a raw JSON string in JSON-object mode.

    Returns:
        The validated ``output_type`` instance.

    Raises:
        BadRequestError: A bad request unrelated to ``response_format`` on the
            strict path is re-raised unchanged, never masked by the fallback.
    """
    try:
        return _validate(await run(build_agent(False)), output_type)
    except BadRequestError as exc:
        if not _rejects_json_schema(exc):
            raise
    return _validate(await run(build_agent(True)), output_type)


def json_object_instructions(
    instructions: str,
    output_type: type[BaseModel],
) -> str:
    """Append a local-validation contract and JSON Schema to agent instructions."""
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


def _rejects_json_schema(exc: BadRequestError) -> bool:
    """Whether a bad request signals an unsupported ``json_schema`` format."""
    message = str(exc).lower()
    return any(marker in message for marker in _RESPONSE_FORMAT_MARKERS)


def _validate(value: object, output_type: type[_OutputT]) -> _OutputT:
    """Validate a final output (raw JSON string or already-parsed) locally."""
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
