"""Natural-language → ``(input, cfg)`` resolver.

``resolve_magic_input`` introspects a flow function's ``input`` and
``cfg`` parameter annotations, builds a parameterized
``ResolvedFlowConfig[InputT, CfgT]``, and runs a lightweight resolver
agent to populate it. The resolver instructions are templated with the
JSON-schema rendering of both types so the model sees exactly which
fields are valid.

This module sits at the top level (not under ``flows/``) because its
output — a ``(input_obj, cfg_obj)`` tuple — is flow-agnostic. The same
resolver works for any future flow that follows the
``(input, *, cfg, ...)`` signature convention.
"""

import inspect
import json
import types
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar, Union, get_args, get_origin

from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel

from quantmind.configs.base import BaseFlowCfg

InputT = TypeVar("InputT", bound=BaseModel)
CfgT = TypeVar("CfgT", bound=BaseModel)


class ResolvedFlowConfig(BaseModel, Generic[InputT, CfgT]):
    """Output schema returned by the resolver agent."""

    input_obj: InputT
    cfg_obj: CfgT


_RESOLVER_INSTRUCTIONS = """\
You are a configuration resolver for the QuantMind {flow_name} flow.
Given a natural-language description of intent, produce a
``ResolvedFlowConfig`` with two fields:

- ``input_obj`` — one variant of the input discriminated union.
- ``cfg_obj``  — the flow configuration.

Rules:
- Set fields conservatively. Leave unspecified fields at their defaults
  rather than inventing values.
- The ``input_obj.type`` discriminator decides which variant you produce.
- Never invent file paths or URLs. If the description does not give a
  concrete identifier, prefer the ``RawText`` variant (when available)
  with the description's content.

Input schema:
{input_schema}

Cfg schema:
{cfg_schema}
"""


async def resolve_magic_input(
    natural_language: str,
    *,
    target_flow: Callable[..., Awaitable[Any]],
    resolver_model: str = "gpt-4o-mini",
    resolver_instructions: str | None = None,
) -> tuple[Any, Any]:
    """Parse ``natural_language`` into ``(input_obj, cfg_obj)`` for ``target_flow``.

    Args:
        natural_language: User-supplied free-form description of intent.
        target_flow: The flow function to resolve for. Must accept
            ``input`` (positional) and ``cfg`` (keyword) parameters.
        resolver_model: LLM used by the resolver agent.
        resolver_instructions: Optional override for the resolver's
            system prompt template. Receives ``flow_name``,
            ``input_schema``, and ``cfg_schema`` via ``str.format``.

    Returns:
        Tuple of ``(input_obj, cfg_obj)`` populated by the resolver.
    """
    input_type, cfg_type = _introspect_flow_signature(target_flow)
    template = resolver_instructions or _RESOLVER_INSTRUCTIONS
    instructions = template.format(
        flow_name=target_flow.__name__,
        input_schema=_pydantic_schema_str(input_type),
        cfg_schema=_pydantic_schema_str(cfg_type),
    )
    resolver: Agent[Any] = Agent(
        name=f"magic_resolver_{target_flow.__name__}",
        instructions=instructions,
        model=resolver_model,
        output_type=AgentOutputSchema(
            # Two layers, both required: model_settings is SkipJsonSchema'd in
            # BaseFlowCfg so the schema builds at all (ModelSettings has callable
            # fields), and strict_json_schema=False accepts the
            # additionalProperties the discriminated-union + knowledge models
            # emit (same rationale as flows/paper.py).
            ResolvedFlowConfig[input_type, cfg_type],  # type: ignore[valid-type]
            strict_json_schema=False,
        ),
    )
    result = await Runner.run(resolver, natural_language)
    out = result.final_output
    return out.input_obj, out.cfg_obj


async def preview_resolve(
    natural_language: str,
    *,
    target_flow: Callable[..., Awaitable[Any]],
    resolver_model: str = "gpt-4o-mini",
) -> tuple[Any, Any]:
    """Resolve and pretty-print the result without invoking the flow."""
    inp, cfg = await resolve_magic_input(
        natural_language,
        target_flow=target_flow,
        resolver_model=resolver_model,
    )
    print("input_obj:", inp.model_dump_json(indent=2))
    print("cfg_obj:", cfg.model_dump_json(indent=2))
    return inp, cfg


def _introspect_flow_signature(
    flow_fn: Callable[..., Any],
) -> tuple[Any, type[BaseFlowCfg]]:
    """Return ``(input_annotation, cfg_type)`` for a flow function.

    ``input_annotation`` is returned as-is — it may be a discriminated-
    union alias such as ``Annotated[Union[...], Field(discriminator=...)]``.
    Pydantic accepts both plain ``BaseModel`` subclasses and discriminated
    aliases as generic parameters.

    ``cfg_type`` strips an outer ``T | None`` so the resolver instantiates
    the concrete cfg subclass. The result must be a ``BaseFlowCfg``
    subclass; anything else means the flow's signature is misshapen.
    """
    sig = inspect.signature(flow_fn)
    if "input" not in sig.parameters:
        raise TypeError(
            f"Flow {flow_fn.__name__!r} must accept an `input` parameter"
        )
    if "cfg" not in sig.parameters:
        raise TypeError(
            f"Flow {flow_fn.__name__!r} must accept a `cfg` keyword parameter"
        )
    input_anno = sig.parameters["input"].annotation
    cfg_anno = sig.parameters["cfg"].annotation
    cfg_type = _strip_optional(cfg_anno)
    if not (isinstance(cfg_type, type) and issubclass(cfg_type, BaseFlowCfg)):
        raise TypeError(
            f"Flow {flow_fn.__name__!r} `cfg` annotation must resolve to "
            f"a BaseFlowCfg subclass (got {cfg_anno!r})"
        )
    return input_anno, cfg_type


def _strip_optional(anno: Any) -> Any:
    """Peel ``T | None`` / ``Optional[T]`` to return the inner T."""
    origin = get_origin(anno)
    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(anno) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return anno


def _pydantic_schema_str(t: Any) -> str:
    """Render a JSON-schema-ish description for resolver instructions.

    Cases handled:

    1. ``Annotated[X, ...]`` — peel via ``__metadata__`` and recurse on X.
    2. Plain ``BaseModel`` subclass — use ``model_json_schema()``.
    3. ``Union[...]`` / ``T | U`` — recurse on each variant; emit
       ``{"oneOf": [...]}``.
    4. Anything else — fall back to ``repr`` so the resolver still gets
       *some* hint. Should not happen for the supported flows.
    """
    if hasattr(t, "__metadata__"):
        inner = get_args(t)[0]
        return _pydantic_schema_str(inner)

    if isinstance(t, type) and hasattr(t, "model_json_schema"):
        try:
            return json.dumps(t.model_json_schema(), indent=2)
        except Exception:
            # Some Pydantic models (e.g. those holding callable fields
            # like ``ModelSettings`` from the agents SDK) cannot render
            # a full JSON schema. Fall back to a name+fields summary
            # so the resolver still has something to work with.
            fields = {
                name: repr(field.annotation)
                for name, field in t.model_fields.items()
            }
            return json.dumps({"title": t.__name__, "fields": fields}, indent=2)

    origin = get_origin(t)
    if origin in (Union, types.UnionType):
        variants = get_args(t)
        schemas = [
            json.loads(_pydantic_schema_str(v))
            for v in variants
            if isinstance(v, type) and hasattr(v, "model_json_schema")
        ]
        return json.dumps({"oneOf": schemas}, indent=2)
    return repr(t)
