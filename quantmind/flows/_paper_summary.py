"""Bounded Agents SDK synthesis for one paper chunk-set artifact."""

import asyncio
import hashlib
import json
from dataclasses import replace
from typing import Any, Protocol

from agents import Agent, ModelSettings, function_tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantmind.configs import PaperFlowCfg
from quantmind.flows._runner import run_with_observability
from quantmind.knowledge import PaperChunkSet, PaperSourceRevision

_SUMMARY_INSTRUCTIONS = """\
Write one accurate global summary of the supplied paper. Use
`read_chunk_group` to inspect source chunks before writing. Cover the central
contribution, architecture or methodology, principal results, and important
limitations. Return only summary prose plus citations. Each citation uses the
zero-based chunk index and a physical page owned by that chunk. Never invent
IDs, source metadata, storage links, pages, or quotations. Read useful
consecutive chunks in groups of up to eight; do not spend one call per chunk.
"""


class PaperSummaryBudgetExceeded(RuntimeError):
    """The summary exceeded a configured runtime or usage boundary."""


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
        if not value.strip():
            raise ValueError("paper summary draft must not be blank")
        return value


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


def _estimated_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class _SummaryBudget:
    """Concurrency-safe accounting for adaptive chunk reads."""

    def __init__(self, cfg: PaperFlowCfg, initial_text: str) -> None:
        initial_tokens = _estimated_tokens(initial_text)
        if initial_tokens > cfg.max_summary_input_tokens:
            raise PaperSummaryBudgetExceeded(
                "paper summary manifest exceeds max_summary_input_tokens"
            )
        self._max_calls = cfg.max_summary_tool_calls
        self._max_input_tokens = cfg.max_summary_input_tokens
        self._calls = 0
        self._input_tokens = initial_tokens
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(cfg.max_summary_concurrency)

    async def read(
        self,
        chunk_set: PaperChunkSet,
        *,
        start: int,
        count: int,
    ) -> str:
        """Return one bounded group while reserving calls and input tokens."""
        if start < 0 or count < 1 or count > 8:
            raise ValueError(
                "start must be non-negative and count must be 1..8"
            )
        if start >= len(chunk_set.chunks):
            raise ValueError("start is outside the chunk-set manifest")
        async with self._semaphore:
            selected = chunk_set.chunks[start : start + count]
            payload = json.dumps(
                [
                    {
                        "chunk_index": chunk.position,
                        "pages": sorted(
                            {span.page_number for span in chunk.source_spans}
                        ),
                        "text": chunk.text,
                    }
                    for chunk in selected
                ],
                ensure_ascii=False,
            )
            tokens = _estimated_tokens(payload)
            async with self._lock:
                if self._calls >= self._max_calls:
                    raise PaperSummaryBudgetExceeded(
                        "paper summary exceeded max_summary_tool_calls"
                    )
                if self._input_tokens + tokens > self._max_input_tokens:
                    raise PaperSummaryBudgetExceeded(
                        "paper summary exceeded max_summary_input_tokens"
                    )
                self._calls += 1
                self._input_tokens += tokens
            return payload


def _summary_manifest(
    source: PaperSourceRevision,
    chunk_set: PaperChunkSet,
    cfg: PaperFlowCfg,
) -> str:
    manifest = [
        {
            "chunk_index": chunk.position,
            "pages": sorted({span.page_number for span in chunk.source_spans}),
            "characters": len(chunk.text),
            "preview": chunk.text[:160],
        }
        for chunk in chunk_set.chunks
    ]
    return json.dumps(
        {
            "title": source.title,
            "authors": source.authors,
            "page_count": len(source.parsed.pages),
            "required_citations": cfg.min_summary_citations,
            "required_source_pages": cfg.min_summary_pages,
            "chunks": manifest,
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
    return hashlib.sha256(
        _summary_instructions(cfg).encode("utf-8")
    ).hexdigest()


def _bounded_model_settings(cfg: PaperFlowCfg) -> ModelSettings:
    settings = cfg.model_settings or ModelSettings()
    configured = settings.max_tokens or cfg.max_summary_output_tokens
    return replace(
        settings,
        max_tokens=min(configured, cfg.max_summary_output_tokens),
        parallel_tool_calls=True,
    )


class _AgentsPaperSummaryProvider:
    """Use one SDK agent with bounded adaptive access to chunk groups."""

    async def summarize(
        self,
        source: PaperSourceRevision,
        chunk_set: PaperChunkSet,
        *,
        cfg: PaperFlowCfg,
    ) -> PaperSummaryDraft:
        manifest = _summary_manifest(source, chunk_set, cfg)
        budget = _SummaryBudget(cfg, manifest)

        @function_tool
        async def read_chunk_group(start: int, count: int) -> str:
            """Read up to eight consecutive paper chunks.

            Args:
                start: Zero-based first chunk index.
                count: Number of chunks to read, from one through eight.
            """
            return await budget.read(
                chunk_set,
                start=start,
                count=count,
            )

        agent: Agent[Any] = Agent(
            name="paper_global_summarizer",
            instructions=_summary_instructions(cfg),
            model=cfg.model,
            model_settings=_bounded_model_settings(cfg),
            tools=[read_chunk_group],
            output_type=PaperSummaryDraft,
        )
        try:
            output = await asyncio.wait_for(
                run_with_observability(
                    agent,
                    manifest,
                    cfg=cfg,
                    memory=None,
                    extra_run_hooks=[],
                ),
                timeout=cfg.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise PaperSummaryBudgetExceeded(
                "paper summary exceeded timeout_seconds"
            ) from exc
        draft = PaperSummaryDraft.model_validate(output)
        if _estimated_tokens(draft.summary) > cfg.max_summary_output_tokens:
            raise PaperSummaryBudgetExceeded(
                "paper summary exceeded max_summary_output_tokens"
            )
        return draft
