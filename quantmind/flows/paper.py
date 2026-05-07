"""Paper extraction flow.

`paper_flow` ingests one of the ``PaperInput`` discriminated-union
variants, fetches and converts the raw payload to markdown via
``preprocess.fetch`` + ``preprocess.format``, then runs an
``Agent(output_type=Paper)`` to produce a typed ``Paper``
``TreeKnowledge`` object.

Customization happens through the configured ``PaperFlowCfg`` (Layer 1)
or the keyword arguments on this function (Layer 2). To swap the whole
flow, fork this file (Layer 3 — design doc §9).
"""

from typing import Any, TypeVar

from agents import Agent, RunHooks, Tool

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import (
    ArxivIdentifier,
    DoiIdentifier,
    HttpUrl,
    LocalFilePath,
    PaperInput,
    RawText,
)
from quantmind.flows._runner import run_with_observability
from quantmind.knowledge import Paper
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
    memory: object | None = None,
    extra_run_hooks: list[RunHooks[Any]] | None = None,
    extra_input_guardrails: list[Any] | None = None,
    extra_output_guardrails: list[Any] | None = None,
) -> P | Paper:
    """Extract a ``Paper`` from a typed ``PaperInput``.

    See design doc §4.1 for the rationale on each kwarg. ``memory`` is a
    PR6 placeholder — non-None values are accepted but unused in PR5.

    Raises:
        UnsupportedContentTypeError: When fetched bytes are not PDF /
            HTML / markdown / plain-text.
        NotImplementedError: When ``input`` is a ``DoiIdentifier`` (the
            unpaywall fallback is its own follow-up issue).
    """
    cfg = cfg or PaperFlowCfg()
    out_type: type[Paper] = output_type or Paper  # type: ignore[assignment]

    raw_md, source_meta = await _fetch_and_format(input)

    # Agent's `model_settings` parameter is non-optional (defaults to a
    # fresh ``ModelSettings()``); only forward when cfg has one set.
    agent_kwargs: dict[str, Any] = {
        "name": "paper_extractor",
        "instructions": _compose_instructions(
            _DEFAULT_INSTRUCTIONS, extra_instructions, cfg
        ),
        "model": cfg.model,
        "tools": list(extra_tools or []),
        "output_type": out_type,
        "input_guardrails": list(extra_input_guardrails or []),
        "output_guardrails": list(extra_output_guardrails or []),
    }
    if cfg.model_settings is not None:
        agent_kwargs["model_settings"] = cfg.model_settings
    agent: Agent[Any] = Agent(**agent_kwargs)
    return await run_with_observability(
        agent,
        _format_input(raw_md, source_meta),
        cfg=cfg,
        memory=memory,
        extra_run_hooks=list(extra_run_hooks or []),
    )


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
