"""Paper extraction flow.

`paper_flow` ingests one of the ``PaperInput`` discriminated-union
variants, fetches and converts the raw payload to markdown via
``preprocess.fetch`` + ``preprocess.format``, then runs an
``Agent(output_type=Paper)`` to produce a typed ``Paper``
``TreeKnowledge`` object.

Customization happens through the configured ``PaperFlowCfg`` (Layer 1)
or the keyword arguments on this function (Layer 2). To swap the whole
flow, fork this file (Layer 3).
"""

import dataclasses
import json
from typing import Any, TypeVar, cast

from agents import Agent, AgentOutputSchema, ModelSettings, RunHooks, Tool

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import (
    ArxivIdentifier,
    DoiIdentifier,
    HttpUrl,
    LocalFilePath,
    PaperInput,
    RawText,
)
from quantmind.flows._providers import configure_provider, provider_capabilities
from quantmind.flows._runner import run_with_observability
from quantmind.knowledge import Paper
from quantmind.mind.memory import Memory
from quantmind.preprocess.fetch import (
    Fetched,
    fetch_arxiv,
    fetch_url,
    read_local_file,
)
from quantmind.preprocess.format import html_to_markdown, pdf_to_markdown

P = TypeVar("P", bound=Paper)

_DEFAULT_INSTRUCTIONS = """\
You are extracting a research paper into a structured QuantMind ``Paper``
TreeKnowledge object. Build the section tree top-down: every node has a
title and a short summary; leaf nodes additionally carry the section
markdown content. Cite supporting passages on each node.

Honour these flags from the run config:
- extract_methodology={extract_methodology}: when true, every methodology
  section becomes its own subtree with a per-step summary.
- extract_limitations={extract_limitations}: when true, surface
  limitations as a dedicated top-level child rather than inlining them.
- asset_class_hint={asset_class_hint!r}: when set, prefer this asset
  class for ``Paper.asset_classes`` if the paper does not state one
  explicitly.

Set ``as_of`` to the publication date when given; otherwise use today's
date. Set the ``source`` provenance ref using the metadata supplied in
the prompt.
"""


class UnsupportedContentTypeError(ValueError):
    """Fetched bytes have a content type paper_flow cannot route to a parser."""


async def paper_flow(
    input: PaperInput,
    *,
    cfg: PaperFlowCfg | None = None,
    extra_tools: list[Tool] | None = None,
    extra_instructions: str | None = None,
    output_type: type[P] | None = None,
    memory: Memory | None = None,
    extra_run_hooks: list[RunHooks[Any]] | None = None,
    extra_input_guardrails: list[Any] | None = None,
    extra_output_guardrails: list[Any] | None = None,
) -> P | Paper:
    """Extract a ``Paper`` from a typed ``PaperInput``.

    When ``memory`` is supplied, ``memory.mcp_servers()`` and
    ``memory.tools()`` flow through to the Agent unconditionally;
    trajectory archiving is gated separately by
    ``cfg.archive_trajectory`` inside the runner.

    Raises:
        UnsupportedContentTypeError: When fetched bytes are not PDF /
            HTML / markdown / plain-text.
        NotImplementedError: When ``input`` is a ``DoiIdentifier`` (the
            unpaywall fallback is its own follow-up issue).
    """
    cfg = cfg or PaperFlowCfg()
    out_type: type[Paper] = output_type or Paper  # type: ignore[assignment]

    raw_md, source_meta = await _fetch_and_format(input)

    # Resolve the provider from cfg.model so the user only needs to set
    # the model string (e.g., "deepseek-chat" vs "gpt-4o") — no manual
    # client / api / tracing setup required at the call site.
    agent_model, cfg = configure_provider(cfg)
    caps = provider_capabilities(cfg.model)

    mcp_servers = memory.mcp_servers() if memory is not None else []
    memory_tools = memory.tools() if memory is not None else []

    base_instructions = _compose_instructions(
        _DEFAULT_INSTRUCTIONS, extra_instructions, cfg
    )

    # Two output-handling paths:
    #
    # 1. supports_json_schema=True  (OpenAI, ...):
    #      AgentOutputSchema drives ``response_format={"type":"json_schema"}``.
    #      The SDK validates against the schema server-side and returns a
    #      Pydantic instance directly.
    #
    # 2. supports_json_schema=False (DeepSeek, ...):
    #      Provider only accepts ``{"type":"json_object"}`` JSON mode. We
    #      embed the Pydantic schema text in the instructions, set
    #      ``model_settings.extra_body.response_format`` to json_object,
    #      keep output_type=None so the Agent returns the raw string, and
    #      validate locally with ``out_type.model_validate_json``.
    if caps.supports_json_schema:
        instructions = base_instructions
        # Non-strict still gives Pydantic validation on the SDK side
        # without forcing every field to be required.
        output_type_arg: Any = AgentOutputSchema(
            out_type, strict_json_schema=False
        )
        effective_model_settings = cfg.model_settings
    else:
        instructions = (
            base_instructions + "\n\nIMPORTANT — output format:\n"
            "Return ONLY a single JSON object (no prose, no markdown "
            "fences) that conforms to this JSON Schema. Every field "
            "marked as `format: uuid` MUST be a real RFC 4122 UUID "
            "(e.g., `12345678-1234-5678-1234-567812345678`); do NOT "
            "emit placeholder strings like `root-uuid-here`. Every "
            "field marked as `format: date-time` MUST be an ISO 8601 "
            "timestamp.\n\n"
            f"```json\n{json.dumps(out_type.model_json_schema(), ensure_ascii=False)}\n```\n"
        )
        output_type_arg = None  # raw string return; we parse below
        # Merge json_object response_format into any user-provided
        # model_settings.extra_body.
        base_ms = cfg.model_settings or ModelSettings()
        # ``extra_body`` is typed as ``httpx.Body`` (a narrow union) but
        # the SDK forwards it as kwargs to AsyncOpenAI's request layer
        # which accepts any JSON-serialisable dict. We cast to keep the
        # branch type-clean without loosening the SDK's signature.
        existing_extra: dict[str, Any] = dict(
            cast(dict[str, Any], base_ms.extra_body or {})
        )
        existing_extra["response_format"] = {"type": "json_object"}
        effective_model_settings = dataclasses.replace(
            base_ms, extra_body=cast(Any, existing_extra)
        )

    agent_kwargs: dict[str, Any] = {
        "name": "paper_extractor",
        "instructions": instructions,
        "model": agent_model,
        "tools": [*(extra_tools or []), *memory_tools],
        "mcp_servers": mcp_servers,
        "input_guardrails": list(extra_input_guardrails or []),
        "output_guardrails": list(extra_output_guardrails or []),
    }
    if output_type_arg is not None:
        agent_kwargs["output_type"] = output_type_arg
    if effective_model_settings is not None:
        agent_kwargs["model_settings"] = effective_model_settings
    agent: Agent[Any] = Agent(**agent_kwargs)
    result = await run_with_observability(
        agent,
        _format_input(raw_md, source_meta),
        cfg=cfg,
        memory=memory,
        extra_run_hooks=list(extra_run_hooks or []),
    )

    if caps.supports_json_schema:
        return result
    # json_object path: result is a raw string; parse + validate.
    return out_type.model_validate_json(_extract_json(result))


async def _fetch_and_format(
    input: PaperInput,
) -> tuple[str, dict[str, Any]]:
    """Dispatch on the input variant; return (markdown, source metadata)."""
    if isinstance(input, ArxivIdentifier):
        raw = await fetch_arxiv(input.id)
        md = await pdf_to_markdown(raw.bytes)
        return md, {
            "source": "arxiv",
            "arxiv_id": raw.arxiv_id,
            "title": raw.title,
            "authors": list(raw.authors),
        }
    if isinstance(input, HttpUrl):
        raw = await fetch_url(input.url)
        md = await _format_by_content_type(raw)
        return md, {
            "source": "web",
            "url": input.url,
            "content_type": raw.content_type,
        }
    if isinstance(input, LocalFilePath):
        raw = await read_local_file(input.path)
        md = await _format_by_content_type(raw)
        return md, {
            "source": "local",
            "path": str(input.path),
            "content_type": raw.content_type,
        }
    if isinstance(input, RawText):
        return input.text, {"source": "inline"}
    if isinstance(input, DoiIdentifier):
        # PR4's CrossrefMetadata exposes only `primary_url` (publisher
        # landing page), not a direct PDF link. Adding the unpaywall
        # fallback that turns a DOI into an OA PDF URL is its own
        # follow-up issue.
        raise NotImplementedError(
            "DOI inputs require an OA PDF resolver (unpaywall fallback) "
            "which is tracked as a PR4 follow-up. Use ArxivIdentifier or "
            "HttpUrl for now."
        )
    raise TypeError(f"Unsupported PaperInput variant: {type(input)!r}")


async def _format_by_content_type(raw: Fetched) -> str:
    """Route a ``Fetched`` payload through the right format helper."""
    ct = (raw.content_type or "").lower()
    if ct.startswith("application/pdf"):
        return await pdf_to_markdown(raw.bytes)
    if ct.startswith("text/html"):
        return await html_to_markdown(
            raw.bytes.decode("utf-8", errors="replace")
        )
    if ct.startswith("text/markdown") or ct.startswith("text/plain"):
        return raw.bytes.decode("utf-8", errors="replace")
    raise UnsupportedContentTypeError(
        f"Unsupported content-type for paper input: {ct!r}"
    )


def _compose_instructions(
    base: str, extra: str | None, cfg: PaperFlowCfg
) -> str:
    """Render the system instructions, appending ``extra`` if provided."""
    instructions = base.format(
        extract_methodology=cfg.extract_methodology,
        extract_limitations=cfg.extract_limitations,
        asset_class_hint=cfg.asset_class_hint,
    )
    if extra:
        instructions = f"{instructions}\n\nAdditional instructions:\n{extra}"
    return instructions


def _format_input(raw_md: str, source_meta: dict[str, Any]) -> str:
    """Concatenate metadata + content into the prompt the agent sees."""
    lines: list[str] = []
    for key, value in source_meta.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(map(str, value))
        lines.append(f"{key}: {value}")
    header = "\n".join(lines)
    return (
        f"--- Source metadata ---\n{header}\n\n--- Paper content ---\n{raw_md}"
    )


def _extract_json(raw: str) -> str:
    r"""Strip optional markdown fences around a JSON payload.

    Even when the system prompt forbids markdown fences, some models
    still wrap their output in ``\`\`\`json ... \`\`\``. We strip that
    one common case so ``model_validate_json`` does not choke on it;
    everything else is left untouched and Pydantic surfaces a clear
    error if the payload is still not valid JSON.
    """
    s = raw.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    # Drop the opening fence (``` or ```json).
    lines = lines[1:]
    # Drop the closing fence, if present.
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
