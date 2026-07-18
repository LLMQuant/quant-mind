"""Bounded multi-agent synthesis for one paper chunk-set artifact."""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, Literal, Protocol

from agents import Agent, FunctionTool, ModelSettings
from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantmind.configs import PaperFlowCfg
from quantmind.flows._runner import run_with_observability
from quantmind.knowledge import PaperChunkSet, PaperSourceRevision

_MAX_RESEARCH_GROUP_SIZE = 8
_ORCHESTRATION_VERSION = "manager-research-agents-v1"

_SUMMARY_INSTRUCTIONS = """\
Act as the paper-summary coordinator. Delegate every suggested research group
to `research_chunk_group` before writing the final summary. You may call
independent groups in parallel and may make bounded overlapping follow-up calls
when a worker report exposes an ambiguity. Synthesize the worker reports into
one accurate global summary covering the central contribution, architecture or
methodology, principal results, and important limitations. Return only summary
prose plus citations. Each citation uses the zero-based chunk index and a
physical page owned by that chunk. Never invent IDs, source metadata, storage
links, pages, or quotations. Set a final citation quote to null unless it is an
exact contiguous substring copied character-for-character from the chunk.
"""

_RESEARCH_INSTRUCTIONS = """\
Act as a bounded paper research specialist. Analyze only the supplied chunk
range and requested focus. Return a concise scope summary plus structured
findings. Classify every finding as context, contribution, method, result, or
limitation and support it with a chunk index and physical page. Do not infer
canonical IDs, source metadata, or claims that are not supported by the
supplied chunks.
"""


class PaperSummaryBudgetExceeded(RuntimeError):
    """The summary exceeded a configured runtime or usage boundary."""


class PaperSummaryCitationDraft(BaseModel):
    """Model-owned citation coordinates before code resolves canonical IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    quote: str | None = Field(default=None, max_length=500)


class PaperResearchRequest(BaseModel):
    """Structured coordinator request for one bounded research subagent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0)
    count: int = Field(ge=1, le=_MAX_RESEARCH_GROUP_SIZE)
    focus: str = Field(min_length=1, max_length=500)

    @field_validator("focus")
    @classmethod
    def _focus_is_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("paper research focus must not be blank")
        return stripped


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


def _estimated_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _research_payload(
    source: PaperSourceRevision,
    chunk_set: PaperChunkSet,
    request: PaperResearchRequest,
) -> str:
    if request.start >= len(chunk_set.chunks):
        raise ValueError("research start is outside the chunk-set manifest")
    selected = chunk_set.chunks[request.start : request.start + request.count]
    if len(selected) != request.count:
        raise ValueError("research range exceeds the chunk-set manifest")
    return json.dumps(
        {
            "title": source.title,
            "authors": source.authors,
            "focus": request.focus,
            "start": request.start,
            "count": request.count,
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


def _coerce_research_draft(value: Any) -> PaperResearchDraft:
    if isinstance(value, str):
        return PaperResearchDraft.model_validate_json(value)
    return PaperResearchDraft.model_validate(value)


def _validate_research_draft(
    chunk_set: PaperChunkSet,
    request: PaperResearchRequest,
    draft: PaperResearchDraft,
) -> None:
    allowed = set(range(request.start, request.start + request.count))
    for finding in draft.findings:
        citation = finding.citation
        if citation.chunk_index not in allowed:
            raise ValueError("research finding cites a chunk outside its scope")
        chunk = chunk_set.chunks[citation.chunk_index]
        pages = {span.page_number for span in chunk.source_spans}
        if citation.page_number not in pages:
            raise ValueError("research finding cites a page outside its chunk")


class _SummaryBudget:
    """Concurrency-safe accounting for nested research-agent calls."""

    def __init__(
        self,
        cfg: PaperFlowCfg,
        initial_text: str,
        *,
        chunk_count: int,
    ) -> None:
        initial_tokens = _estimated_tokens(initial_text)
        if initial_tokens > cfg.max_summary_input_tokens:
            raise PaperSummaryBudgetExceeded(
                "paper summary manifest exceeds max_summary_input_tokens"
            )
        required_calls = (
            chunk_count + _MAX_RESEARCH_GROUP_SIZE - 1
        ) // _MAX_RESEARCH_GROUP_SIZE
        if required_calls > cfg.max_summary_tool_calls:
            raise PaperSummaryBudgetExceeded(
                "max_summary_tool_calls cannot cover every paper chunk"
            )
        self._max_calls = cfg.max_summary_tool_calls
        self._max_input_tokens = cfg.max_summary_input_tokens
        self._max_worker_output_tokens = cfg.max_summary_worker_output_tokens
        self._max_final_output_tokens = cfg.max_summary_output_tokens
        self._max_total_output_tokens = cfg.max_summary_total_output_tokens
        self._calls = 0
        self._input_tokens = initial_tokens
        self._output_tokens = 0
        self._reserved_input_tokens = 0
        self._reserved_output_tokens = 0
        self._covered_chunks: set[int] = set()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(cfg.max_summary_concurrency)

    @property
    def research_calls(self) -> int:
        """Return the number of research calls reserved so far."""
        return self._calls

    @property
    def covered_chunks(self) -> frozenset[int]:
        """Return chunk positions covered by successful worker reports."""
        return frozenset(self._covered_chunks)

    async def invoke_research(
        self,
        source: PaperSourceRevision,
        chunk_set: PaperChunkSet,
        request: PaperResearchRequest,
        operation: Callable[[], Awaitable[Any]],
    ) -> str:
        """Run one agent tool call while reserving all shared budgets."""
        payload = _research_payload(source, chunk_set, request)
        payload_tokens = _estimated_tokens(payload)
        input_reservation = payload_tokens + self._max_worker_output_tokens
        await self._semaphore.acquire()
        reserved = False
        try:
            async with self._lock:
                if self._calls >= self._max_calls:
                    raise PaperSummaryBudgetExceeded(
                        "paper summary exceeded max_summary_tool_calls"
                    )
                projected_input = (
                    self._input_tokens
                    + self._reserved_input_tokens
                    + input_reservation
                )
                if projected_input > self._max_input_tokens:
                    raise PaperSummaryBudgetExceeded(
                        "paper summary exceeded max_summary_input_tokens"
                    )
                projected_output = (
                    self._output_tokens
                    + self._reserved_output_tokens
                    + self._max_worker_output_tokens
                    + self._max_final_output_tokens
                )
                if projected_output > self._max_total_output_tokens:
                    raise PaperSummaryBudgetExceeded(
                        "paper summary exceeded max_summary_total_output_tokens"
                    )
                self._calls += 1
                self._reserved_input_tokens += input_reservation
                self._reserved_output_tokens += self._max_worker_output_tokens
                reserved = True

            raw_output = await operation()
            draft = _coerce_research_draft(raw_output)
            _validate_research_draft(chunk_set, request, draft)
            serialized = draft.model_dump_json()
            output_tokens = _estimated_tokens(serialized)
            if output_tokens > self._max_worker_output_tokens:
                raise PaperSummaryBudgetExceeded(
                    "paper research worker exceeded "
                    "max_summary_worker_output_tokens"
                )

            async with self._lock:
                self._reserved_input_tokens -= input_reservation
                self._reserved_output_tokens -= self._max_worker_output_tokens
                self._input_tokens += payload_tokens + output_tokens
                self._output_tokens += output_tokens
                self._covered_chunks.update(
                    range(request.start, request.start + request.count)
                )
                reserved = False
            return serialized
        finally:
            if reserved:
                async with self._lock:
                    self._reserved_input_tokens -= input_reservation
                    self._reserved_output_tokens -= (
                        self._max_worker_output_tokens
                    )
            self._semaphore.release()

    async def accept_final_output(self, draft: PaperSummaryDraft) -> None:
        """Validate final output and require complete worker coverage."""
        serialized = draft.model_dump_json()
        output_tokens = _estimated_tokens(serialized)
        if output_tokens > self._max_final_output_tokens:
            raise PaperSummaryBudgetExceeded(
                "paper summary exceeded max_summary_output_tokens"
            )
        async with self._lock:
            if (
                self._output_tokens + output_tokens
                > self._max_total_output_tokens
            ):
                raise PaperSummaryBudgetExceeded(
                    "paper summary exceeded max_summary_total_output_tokens"
                )
            self._output_tokens += output_tokens

    def require_complete_coverage(self, chunk_count: int) -> None:
        """Reject a coordinator that did not delegate every source chunk."""
        missing = set(range(chunk_count)) - self._covered_chunks
        if missing:
            preview = ", ".join(str(index) for index in sorted(missing)[:8])
            raise PaperSummaryBudgetExceeded(
                "paper summary research did not cover every chunk; missing "
                f"{preview}"
            )


def _suggested_research_groups(chunk_count: int) -> list[dict[str, int]]:
    return [
        {
            "start": start,
            "count": min(_MAX_RESEARCH_GROUP_SIZE, chunk_count - start),
        }
        for start in range(0, chunk_count, _MAX_RESEARCH_GROUP_SIZE)
    ]


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
            "required_research_groups": _suggested_research_groups(
                len(chunk_set.chunks)
            ),
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
    payload = json.dumps(
        {
            "orchestration": _ORCHESTRATION_VERSION,
            "coordinator_instructions": _summary_instructions(cfg),
            "research_instructions": _RESEARCH_INSTRUCTIONS,
            "max_research_calls": cfg.max_summary_tool_calls,
            "max_research_concurrency": cfg.max_summary_concurrency,
            "research_max_turns": cfg.max_summary_worker_turns,
            "research_max_output_tokens": (
                cfg.max_summary_worker_output_tokens
            ),
            "max_total_input_tokens": cfg.max_summary_input_tokens,
            "max_total_output_tokens": cfg.max_summary_total_output_tokens,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bounded_model_settings(cfg: PaperFlowCfg) -> ModelSettings:
    settings = cfg.model_settings or ModelSettings()
    configured = settings.max_tokens or cfg.max_summary_output_tokens
    return replace(
        settings,
        max_tokens=min(configured, cfg.max_summary_output_tokens),
        parallel_tool_calls=True,
    )


def _bounded_research_model_settings(cfg: PaperFlowCfg) -> ModelSettings:
    settings = cfg.model_settings or ModelSettings()
    configured = settings.max_tokens or cfg.max_summary_worker_output_tokens
    return replace(
        settings,
        max_tokens=min(configured, cfg.max_summary_worker_output_tokens),
        parallel_tool_calls=False,
    )


def _build_research_agent_tool(
    source: PaperSourceRevision,
    chunk_set: PaperChunkSet,
    cfg: PaperFlowCfg,
    budget: _SummaryBudget,
) -> FunctionTool:
    """Expose one bounded research agent to the summary coordinator."""
    researcher: Agent[Any] = Agent(
        name="paper_chunk_researcher",
        instructions=_RESEARCH_INSTRUCTIONS,
        model=cfg.model,
        model_settings=_bounded_research_model_settings(cfg),
        output_type=PaperResearchDraft,
    )

    def build_input(options: Any) -> str:
        request = PaperResearchRequest.model_validate(options["params"])
        return _research_payload(source, chunk_set, request)

    async def extract_output(result: Any) -> str:
        draft = PaperResearchDraft.model_validate(result.final_output)
        return draft.model_dump_json()

    tool = researcher.as_tool(
        tool_name="research_chunk_group",
        tool_description=(
            "Delegate one bounded chunk range to a paper research subagent. "
            "Call every required research group before final synthesis and "
            "use extra overlapping calls only for focused follow-up."
        ),
        custom_output_extractor=extract_output,
        max_turns=cfg.max_summary_worker_turns,
        parameters=PaperResearchRequest,
        input_builder=build_input,
        failure_error_function=None,
    )
    invoke_agent = tool.on_invoke_tool

    async def invoke_bounded_agent(context: Any, input_json: str) -> str:
        request = PaperResearchRequest.model_validate_json(input_json)
        return await budget.invoke_research(
            source,
            chunk_set,
            request,
            lambda: invoke_agent(context, input_json),
        )

    tool.on_invoke_tool = invoke_bounded_agent
    return tool


class _AgentsPaperSummaryProvider:
    """Coordinate bounded research subagents and synthesize one summary."""

    async def summarize(
        self,
        source: PaperSourceRevision,
        chunk_set: PaperChunkSet,
        *,
        cfg: PaperFlowCfg,
    ) -> PaperSummaryDraft:
        manifest = _summary_manifest(source, chunk_set, cfg)
        budget = _SummaryBudget(
            cfg,
            manifest,
            chunk_count=len(chunk_set.chunks),
        )
        research_tool = _build_research_agent_tool(
            source,
            chunk_set,
            cfg,
            budget,
        )
        coordinator: Agent[Any] = Agent(
            name="paper_summary_coordinator",
            instructions=_summary_instructions(cfg),
            model=cfg.model,
            model_settings=_bounded_model_settings(cfg),
            tools=[research_tool],
            output_type=PaperSummaryDraft,
        )
        try:
            output = await asyncio.wait_for(
                run_with_observability(
                    coordinator,
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
        budget.require_complete_coverage(len(chunk_set.chunks))
        await budget.accept_final_output(draft)
        return draft
