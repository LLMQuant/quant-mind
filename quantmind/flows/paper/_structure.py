"""Windowed full-text draft structuring for an exact paper source revision."""

import asyncio
import hashlib
import json
from dataclasses import replace
from typing import Any, Literal, Protocol

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

_STRUCTURE_ORCHESTRATION: Literal["windowed-v1"] = "windowed-v1"

_QUALITY_ORDER = {"low": 0, "medium": 1, "high": 2}

_STRUCTURE_INSTRUCTIONS = """\
Act as a paper structure specialist. Return one hierarchy draft and a quality
rating. Use only the supplied outline signals and ordered physical-page text.
Every node must name one inclusive physical-page span; a parent must include
all physical pages included by its children. The payload covers one window of
consecutive pages and names the document's full page range. When a prior
draft is supplied as draft_so_far, extend or revise it with evidence from the
window's pages and return the complete updated hierarchy, keeping earlier
sections unless the new pages contradict them. The returned root must cover
every page read so far; after the final window that is every document page.
Use titles and concise summaries for reasoning. Do not invent UUIDs, parent
links, citations, source text, or canonical identity. If the evidence does not
support a reliable hierarchy, set quality to low so code can build a safe flat
fallback.
"""


class PaperStructureError(RuntimeError):
    """Paper structuring exceeded a configured runtime boundary."""


class _PaperStructureProvider(Protocol):
    """Test seam and production boundary for one structure draft."""

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
            "window_chars": cfg.window_chars,
            "window_overlap_pages": cfg.window_overlap_pages,
            "orchestration": _STRUCTURE_ORCHESTRATION,
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


def _page_payloads(
    source: PaperSourceRevision,
    cfg: PaperStructureCfg,
) -> tuple[dict[str, Any], ...]:
    """Project parsed pages into prompt entries, clipping only when asked."""
    return tuple(
        {
            "page_number": page.page_number,
            "text": (
                page.text
                if cfg.page_text_chars is None
                else page.text[: cfg.page_text_chars]
            ),
        }
        for page in source.parsed.pages
    )


def _window_pages(
    pages: tuple[dict[str, Any], ...],
    *,
    window_chars: int,
    overlap_pages: int,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Split ordered page entries into character-bounded page windows.

    Pages are packed greedily until ``window_chars`` is reached; a page is
    never split, so an oversized page forms its own window. Consecutive
    windows share ``overlap_pages`` trailing pages for continuity, and every
    window starts at least one page after its predecessor so packing always
    terminates.
    """
    windows: list[tuple[dict[str, Any], ...]] = []
    start = 0
    while start < len(pages):
        end = start
        used = 0
        while end < len(pages):
            page_chars = len(pages[end]["text"])
            if end > start and used + page_chars > window_chars:
                break
            used += page_chars
            end += 1
        windows.append(tuple(pages[start:end]))
        if end >= len(pages):
            break
        start = max(end - overlap_pages, start + 1)
    return tuple(windows)


def _structure_payload(
    signals: OutlineSignals,
    pages: tuple[dict[str, Any], ...],
    window: tuple[dict[str, Any], ...],
    *,
    window_index: int,
    window_total: int,
    draft_so_far: PaperStructureTreeDraft | None,
) -> str:
    return json.dumps(
        {
            "document": {
                "first_page": pages[0]["page_number"],
                "last_page": pages[-1]["page_number"],
                "page_count": len(pages),
            },
            "window": {
                "index": window_index + 1,
                "total": window_total,
                "start_page": window[0]["page_number"],
                "end_page": window[-1]["page_number"],
            },
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
            "draft_so_far": (
                None
                if draft_so_far is None
                else draft_so_far.root.model_dump(mode="json")
            ),
            "pages": list(window),
        },
        ensure_ascii=False,
    )


class _AgentsPaperStructureProvider:
    """Draft one hierarchy from full page text in character-bounded windows.

    Every window carries complete page text (optionally clipped by
    ``cfg.page_text_chars``); a document larger than ``cfg.window_chars``
    is drafted across several model calls, each extending the prior draft.
    ``cfg.timeout_seconds`` bounds each model call. The returned draft keeps
    the worst quality rating seen across windows, so one unreliable window
    routes the whole document to the deterministic flat fallback.
    """

    async def structure(
        self,
        signals: OutlineSignals,
        source: PaperSourceRevision,
        *,
        cfg: PaperStructureCfg,
    ) -> PaperStructureTreeDraft:
        pages = _page_payloads(source, cfg)
        if not pages:
            raise PaperStructureError(
                "paper structure drafting requires at least one parsed page"
            )
        windows = _window_pages(
            pages,
            window_chars=cfg.window_chars,
            overlap_pages=cfg.window_overlap_pages,
        )
        draft: PaperStructureTreeDraft | None = None
        worst_quality: Literal["low", "medium", "high"] = "high"
        for window_index, window in enumerate(windows):
            payload = _structure_payload(
                signals,
                pages,
                window,
                window_index=window_index,
                window_total=len(windows),
                draft_so_far=draft,
            )
            draft = await self._draft_window(payload, cfg)
            if _QUALITY_ORDER[draft.quality] < _QUALITY_ORDER[worst_quality]:
                worst_quality = draft.quality
        if draft is None:  # pragma: no cover - guarded by the pages check
            raise PaperStructureError(
                "paper structure drafting produced no draft"
            )
        if draft.quality != worst_quality:
            draft = PaperStructureTreeDraft(
                root=draft.root,
                quality=worst_quality,
            )
        return draft

    async def _draft_window(
        self,
        payload: str,
        cfg: PaperStructureCfg,
    ) -> PaperStructureTreeDraft:
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
