"""Deterministic map-reduce summarization for one paper chunk-set artifact.

The chunk set is tiled into fixed-size groups by code, so every chunk is
covered exactly once. There is no coordinator agent deciding *how* to
decompose the work, and therefore no need to police coverage or reconcile a
shared token budget after the fact. One research agent runs per group (bounded
fan-out), and one reducer agent synthesizes the group reports into a single
cited global summary.

Bounding is delegated to the Agents SDK: per-agent output is capped through
``ModelSettings.max_tokens`` and the reducer is wrapped in ``asyncio.wait_for``.
The removed manager/worker design instead ran an autonomous coordinator and
then fought its nondeterminism with a hand-rolled concurrency-safe accountant;
that is precisely the runtime the SDK already provides.
"""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from agents import Agent, ModelSettings
from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantmind.configs import PaperFlowCfg
from quantmind.flows._runner import run_with_observability
from quantmind.knowledge import PaperChunkSet, PaperSourceRevision

_ORCHESTRATION_VERSION = "map-reduce-v1"

_SUMMARY_INSTRUCTIONS = """\
Act as the paper-summary reducer. You are given research reports that together
cover every chunk of the paper. Synthesize them into one accurate global
summary covering the central contribution, architecture or methodology,
principal results, and important limitations. Return only summary prose plus
citations. Each citation uses the zero-based chunk index and a physical page
reported by a research finding. Set a citation quote to null unless it is
copied verbatim from a research finding's quote for that same chunk. Never
invent chunk indices, pages, or quotations.
"""

_RESEARCH_INSTRUCTIONS = """\
Act as a bounded paper research specialist. Analyze only the supplied chunk
range. Return a concise scope summary plus structured findings. Classify every
finding as context, contribution, method, result, or limitation and support it
with a chunk index and physical page drawn from the supplied chunks. When you
quote, copy an exact contiguous substring from the chunk text; otherwise leave
the quote null. Do not infer canonical IDs or claims unsupported by the chunks.
"""


class PaperSummaryError(RuntimeError):
    """The summary exceeded a configured runtime boundary."""


class PaperResearchCitationDraft(BaseModel):
    """Chunk and page coordinates returned by a research subagent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_index: int = Field(ge=0)
    page_number: int = Field(ge=1)


class PaperResearchFindingDraft(BaseModel):
    """One subagent finding with non-canonical source coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "context",
        "contribution",
        "method",
        "result",
        "limitation",
    ]
    claim: str = Field(min_length=1)
    citation: PaperResearchCitationDraft
    quote: str | None = Field(default=None, max_length=500)

    @field_validator("claim")
    @classmethod
    def _claim_is_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("paper research claim must not be blank")
        return stripped


class PaperResearchDraft(BaseModel):
    """Typed evidence report returned by one research subagent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope_summary: str = Field(min_length=1)
    findings: tuple[PaperResearchFindingDraft, ...] = Field(min_length=1)

    @field_validator("scope_summary")
    @classmethod
    def _scope_summary_is_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("paper research scope summary must not be blank")
        return stripped


class PaperSummaryCitationDraft(BaseModel):
    """Model-owned citation coordinates before code resolves canonical IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    quote: str | None = Field(default=None, max_length=500)


class PaperSummaryDraft(BaseModel):
    """Limited model output containing prose and non-canonical coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1)
    citations: tuple[PaperSummaryCitationDraft, ...] = Field(min_length=1)

    @field_validator("summary")
    @classmethod
    def _summary_is_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("paper summary draft must not be blank")
        return stripped


class _PaperSummaryProvider(Protocol):
    """Deterministic test seam and production summary-provider boundary."""

    async def summarize(
        self,
        source: PaperSourceRevision,
        chunk_set: PaperChunkSet,
        *,
        cfg: PaperFlowCfg,
    ) -> PaperSummaryDraft:
        """Create one bounded draft from the selected chunk set."""
        ...


@dataclass(frozen=True)
class _ChunkGroup:
    """A contiguous, code-chosen range of chunk positions."""

    start: int
    count: int


def _chunk_groups(chunk_count: int, size: int) -> list[_ChunkGroup]:
    """Tile ``[0, chunk_count)`` so every chunk lands in exactly one group."""
    return [
        _ChunkGroup(start=start, count=min(size, chunk_count - start))
        for start in range(0, chunk_count, size)
    ]


def _research_payload(
    source: PaperSourceRevision,
    chunk_set: PaperChunkSet,
    group: _ChunkGroup,
) -> str:
    selected = chunk_set.chunks[group.start : group.start + group.count]
    return json.dumps(
        {
            "title": source.title,
            "authors": source.authors,
            "start": group.start,
            "count": group.count,
            "chunks": [
                {
                    "chunk_index": chunk.position,
                    "pages": sorted(
                        {span.page_number for span in chunk.source_spans}
                    ),
                    "text": chunk.text,
                }
                for chunk in selected
            ],
        },
        ensure_ascii=False,
    )


def _validate_research_draft(
    chunk_set: PaperChunkSet,
    group: _ChunkGroup,
    draft: PaperResearchDraft,
) -> None:
    allowed = set(range(group.start, group.start + group.count))
    for finding in draft.findings:
        citation = finding.citation
        if citation.chunk_index not in allowed:
            raise ValueError("research finding cites a chunk outside its group")
        chunk = chunk_set.chunks[citation.chunk_index]
        pages = {span.page_number for span in chunk.source_spans}
        if citation.page_number not in pages:
            raise ValueError("research finding cites a page outside its chunk")
        if finding.quote is not None and finding.quote not in chunk.text:
            raise ValueError(
                "research finding quote is not present in its chunk"
            )


def _reduce_payload(
    source: PaperSourceRevision,
    chunk_set: PaperChunkSet,
    reports: Sequence[PaperResearchDraft],
) -> str:
    return json.dumps(
        {
            "title": source.title,
            "authors": source.authors,
            "chunk_count": len(chunk_set.chunks),
            "reports": [
                {
                    "scope_summary": report.scope_summary,
                    "findings": [
                        {
                            "kind": finding.kind,
                            "claim": finding.claim,
                            "chunk_index": finding.citation.chunk_index,
                            "page_number": finding.citation.page_number,
                            "quote": finding.quote,
                        }
                        for finding in report.findings
                    ],
                }
                for report in reports
            ],
        },
        ensure_ascii=False,
    )


def _summary_instructions(cfg: PaperFlowCfg) -> str:
    instructions = _SUMMARY_INSTRUCTIONS
    if cfg.summary_instructions:
        instructions = (
            f"{instructions}\n\nAdditional summary requirements:\n"
            f"{cfg.summary_instructions}"
        )
    return instructions


def _summary_instructions_hash(cfg: PaperFlowCfg) -> str:
    payload = json.dumps(
        {
            "orchestration": _ORCHESTRATION_VERSION,
            "reducer_instructions": _summary_instructions(cfg),
            "research_instructions": _RESEARCH_INSTRUCTIONS,
            "research_group_size": cfg.summary_research_group_size,
            "max_output_tokens": cfg.max_summary_output_tokens,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _summary_model_settings(cfg: PaperFlowCfg) -> ModelSettings:
    settings = cfg.model_settings or ModelSettings()
    configured = settings.max_tokens or cfg.max_summary_output_tokens
    return replace(
        settings,
        max_tokens=min(configured, cfg.max_summary_output_tokens),
    )


class _AgentsPaperSummaryProvider:
    """Fan out one research agent per chunk group, then reduce to a summary."""

    async def summarize(
        self,
        source: PaperSourceRevision,
        chunk_set: PaperChunkSet,
        *,
        cfg: PaperFlowCfg,
    ) -> PaperSummaryDraft:
        groups = _chunk_groups(
            len(chunk_set.chunks), cfg.summary_research_group_size
        )
        model_settings = _summary_model_settings(cfg)
        researcher: Agent[Any] = Agent(
            name="paper_chunk_researcher",
            instructions=_RESEARCH_INSTRUCTIONS,
            model=cfg.model,
            model_settings=model_settings,
            output_type=PaperResearchDraft,
        )
        semaphore = asyncio.Semaphore(cfg.summary_concurrency)

        async def study(group: _ChunkGroup) -> PaperResearchDraft:
            async with semaphore:
                output = await run_with_observability(
                    researcher,
                    _research_payload(source, chunk_set, group),
                    cfg=cfg,
                    memory=None,
                    extra_run_hooks=[],
                )
            draft = PaperResearchDraft.model_validate(output)
            _validate_research_draft(chunk_set, group, draft)
            return draft

        reports = await asyncio.gather(*(study(group) for group in groups))

        reducer: Agent[Any] = Agent(
            name="paper_summary_reducer",
            instructions=_summary_instructions(cfg),
            model=cfg.model,
            model_settings=model_settings,
            output_type=PaperSummaryDraft,
        )
        try:
            output = await asyncio.wait_for(
                run_with_observability(
                    reducer,
                    _reduce_payload(source, chunk_set, reports),
                    cfg=cfg,
                    memory=None,
                    extra_run_hooks=[],
                ),
                timeout=cfg.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise PaperSummaryError(
                "paper summary exceeded timeout_seconds"
            ) from exc
        return PaperSummaryDraft.model_validate(output)
