"""Single-pass draft structuring for an exact paper source revision."""

import asyncio
import hashlib
import json
from dataclasses import replace
from typing import Any, Protocol

from agents import Agent, ModelSettings

from quantmind.configs import PaperStructureCfg
from quantmind.flows._runner import run_with_observability
from quantmind.knowledge import PaperSourceRevision, PaperStructureTreeDraft
from quantmind.preprocess import OutlineSignals
from quantmind.utils.structured_output import (
    json_object_instructions,
    json_object_model_settings,
    run_structured,
)

_STRUCTURE_INSTRUCTIONS = """\
Act as a paper structure specialist. Return one hierarchy draft and a quality
rating. Use only the supplied outline signals and ordered physical-page text.
Every node must name one inclusive physical-page span; a parent must include
all physical pages included by its children. The root must cover every page.
Use titles and concise summaries for reasoning. Do not invent UUIDs, parent
links, citations, source text, or canonical identity. If the evidence does not
support a reliable hierarchy, set quality to low so code can build a safe flat
fallback.
"""


class PaperStructureError(RuntimeError):
    """Paper structuring exceeded a configured runtime boundary."""


class _PaperStructureProvider(Protocol):
    """Test seam and production boundary for one structure draft call."""

    async def structure(
        self,
        signals: OutlineSignals,
        source: PaperSourceRevision,
        *,
        cfg: PaperStructureCfg,
    ) -> PaperStructureTreeDraft:
        """Create one bounded hierarchy draft without canonical identity."""
        ...


def _structure_instructions(cfg: PaperStructureCfg) -> str:
    if cfg.instructions is None:
        return _STRUCTURE_INSTRUCTIONS
    return (
        f"{_STRUCTURE_INSTRUCTIONS}\n\nAdditional structure requirements:\n"
        f"{cfg.instructions}"
    )


def _structure_instructions_hash(cfg: PaperStructureCfg) -> str:
    payload = json.dumps(
        {
            "instructions": _structure_instructions(cfg),
            "max_depth": cfg.max_depth,
            "max_nodes": cfg.max_nodes,
            "max_output_tokens": cfg.max_output_tokens,
            "page_text_chars": cfg.page_text_chars,
            "orchestration": "single-pass-v1",
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _structure_model_settings(cfg: PaperStructureCfg) -> ModelSettings:
    settings = cfg.model_settings or ModelSettings()
    configured = settings.max_tokens or cfg.max_output_tokens
    return replace(
        settings,
        max_tokens=min(configured, cfg.max_output_tokens),
    )


def _structure_payload(
    signals: OutlineSignals,
    source: PaperSourceRevision,
    cfg: PaperStructureCfg,
) -> str:
    return json.dumps(
        {
            "outline": {
                "table_of_contents_pages": signals.table_of_contents_pages,
                "printed_page_offset": signals.printed_page_offset,
                "headings": [
                    {
                        "page_number": heading.page_number,
                        "text": heading.text,
                        "level_hint": heading.level_hint,
                    }
                    for heading in signals.headings
                ],
            },
            "pages": [
                {
                    "page_number": page.page_number,
                    "text": page.text[: cfg.page_text_chars],
                }
                for page in source.parsed.pages
            ],
        },
        ensure_ascii=False,
    )


class _AgentsPaperStructureProvider:
    """Run one structured-output agent over deterministic outline signals."""

    async def structure(
        self,
        signals: OutlineSignals,
        source: PaperSourceRevision,
        *,
        cfg: PaperStructureCfg,
    ) -> PaperStructureTreeDraft:
        payload = _structure_payload(signals, source, cfg)

        def build_agent(json_object: bool) -> Agent[Any]:
            instructions = _structure_instructions(cfg)
            model_settings = _structure_model_settings(cfg)
            kwargs: dict[str, Any] = {
                "name": "paper_structure_builder",
                "model": cfg.model,
            }
            if json_object:
                kwargs["instructions"] = json_object_instructions(
                    instructions, PaperStructureTreeDraft
                )
                kwargs["model_settings"] = json_object_model_settings(
                    model_settings
                )
            else:
                kwargs["instructions"] = instructions
                kwargs["model_settings"] = model_settings
                kwargs["output_type"] = PaperStructureTreeDraft
            return Agent(**kwargs)

        async def run_agent(agent: Agent[Any]) -> Any:
            try:
                return await asyncio.wait_for(
                    run_with_observability(
                        agent,
                        payload,
                        cfg=cfg,
                        memory=None,
                        extra_run_hooks=[],
                    ),
                    timeout=cfg.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise PaperStructureError(
                    "paper structure build exceeded timeout_seconds"
                ) from exc

        return await run_structured(
            PaperStructureTreeDraft,
            build_agent=build_agent,
            run=run_agent,
        )
