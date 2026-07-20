"""Vectorless reasoning-based retrieval over a validated structure tree."""

import asyncio
import json
from typing import Protocol, cast, runtime_checkable
from uuid import UUID

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool
from pydantic import BaseModel, ConfigDict, Field

from quantmind.configs import RetrievalCfg
from quantmind.knowledge import (
    ArtifactLocator,
    Citation,
    StructureTree,
    TreeNode,
)
from quantmind.library import LocalKnowledgeLibrary

_SINGLE_PASS_INSTRUCTIONS = """\
Select the structure-tree nodes that contain evidence relevant to the question.
Reason only over the supplied titles, summaries, hierarchy, and optional seed
nodes. Return node UUIDs, not an answer. Prefer the smallest sufficient set and
never invent an ID.
"""

_AGENTIC_INSTRUCTIONS = """\
Retrieve evidence relevant to the question by inspecting the document structure
and opening node content as needed. Return the UUIDs of the smallest sufficient
evidence nodes, not an answer. Never invent an ID. Seed nodes are hints only;
you may inspect other branches when the structure supports it.
"""


class RetrievalError(RuntimeError):
    """Reasoning-based retrieval exceeded a runtime or evidence boundary."""


class RetrievalEvidence(BaseModel):
    """One selected node with lazily resolved page-cited source content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    locator: ArtifactLocator
    title: str
    content: str = Field(min_length=1)
    citations: tuple[Citation, ...] = Field(min_length=1)


class _RetrievalSelectionDraft(BaseModel):
    """Model-selected canonical node coordinates before code resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_ids: tuple[UUID, ...] = Field(min_length=1)


@runtime_checkable
class _ResolvableStructure(Protocol):
    """Identity required to address one StructureTree through a library."""

    id: UUID
    source_revision_id: UUID
    artifact_kind: str


class StructureRetriever:
    """Reusable reasoning service for one structure tree per query.

    The service owns stable library and retrieval policy dependencies. It does
    not retain a current structure, question, seed, or result, so one instance
    can safely query different document trees.
    """

    __slots__ = ("_cfg", "_library")

    def __init__(
        self,
        *,
        library: LocalKnowledgeLibrary,
        cfg: RetrievalCfg,
    ) -> None:
        """Initialize retrieval with shared dependencies and policy.

        Args:
            library: Canonical library used for lazy node resolution.
            cfg: Model, retrieval grain, and structure/evidence bounds.
        """
        self._library = library
        self._cfg = cfg.model_copy(deep=True)

    async def retrieve(
        self,
        structure: StructureTree,
        question: str,
        *,
        seed_locators: list[ArtifactLocator] | None = None,
    ) -> list[RetrievalEvidence]:
        """Select and resolve page-cited evidence from one structure tree.

        Args:
            structure: Validated shared tree with a resolvable artifact binding.
            question: Evidence need used for relevance reasoning.
            seed_locators: Optional same-tree node hints reserved for a later
                semantic shortlist. This operation performs no semantic search.

        Returns:
            Selected node evidence with canonical locators and source content.

        Raises:
            ValueError: If the question, seeds, or selected node IDs are invalid.
            TypeError: If the structure has no resolvable artifact identity.
            RetrievalError: If runtime or evidence bounds are exceeded.
        """
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("retrieval question must not be blank")
        structure.validate()
        identity = _resolvable_identity(structure)
        seed_ids = _validate_seed_locators(
            structure,
            identity,
            seed_locators or [],
        )
        serialized = _serialize_structure(
            structure,
            token_budget=self._cfg.structure_token_budget,
            seed_ids=set(seed_ids),
        )
        if self._cfg.grain == "single-pass":
            selection = await _single_pass_select(
                normalized_question,
                serialized,
                seed_ids,
                self._cfg,
            )
        else:
            selection = await _agentic_select(
                structure,
                identity,
                normalized_question,
                serialized,
                seed_ids,
                self._library,
                self._cfg,
            )
        node_ids = _validate_selection(structure, selection, self._cfg)
        return [
            await _resolve_evidence(
                structure,
                identity,
                node_id,
                self._library,
            )
            for node_id in node_ids
        ]


def _resolvable_identity(structure: StructureTree) -> _ResolvableStructure:
    if not isinstance(structure, _ResolvableStructure):
        raise TypeError(
            "structure must expose id, source_revision_id, and artifact_kind"
        )
    return structure


def _validate_seed_locators(
    structure: StructureTree,
    identity: _ResolvableStructure,
    locators: list[ArtifactLocator],
) -> tuple[UUID, ...]:
    node_ids: list[UUID] = []
    for locator in locators:
        if (
            locator.source_revision_id != identity.source_revision_id
            or locator.artifact_id != identity.id
            or locator.artifact_kind != identity.artifact_kind
            or locator.member_id is None
            or locator.member_id not in structure.nodes
        ):
            raise ValueError(
                "seed locator must address a node in the selected structure tree"
            )
        if locator.member_id not in node_ids:
            node_ids.append(locator.member_id)
    return tuple(node_ids)


def _serialize_structure(
    structure: StructureTree,
    *,
    token_budget: int,
    seed_ids: set[UUID],
) -> str:
    character_budget = token_budget * 4
    payload: dict[str, object] = {
        "root_node_id": str(structure.root_node_id),
        "nodes": [],
        "truncated": False,
    }
    serialized_nodes = cast(list[dict[str, object]], payload["nodes"])
    for node in structure.walk_dfs():
        candidate: dict[str, object] = {
            "node_id": str(node.node_id),
            "parent_id": str(node.parent_id) if node.parent_id else None,
            "position": node.position,
            "title": node.title,
            "summary": node.summary,
            "children_ids": [str(value) for value in node.children_ids],
            "seeded": node.node_id in seed_ids,
        }
        serialized_nodes.append(candidate)
        value = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(value) <= character_budget:
            continue
        candidate["summary"] = node.summary[:80]
        value = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(value) <= character_budget:
            continue
        serialized_nodes.pop()
        payload["truncated"] = True
        break
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _run_config(cfg: RetrievalCfg) -> RunConfig:
    return RunConfig(
        workflow_name=cfg.workflow_name or "quantmind.structure_retrieval",
        trace_metadata=cfg.trace_metadata,
        trace_include_sensitive_data=cfg.trace_include_sensitive_data,
        tracing_disabled=cfg.tracing_disabled,
    )


async def _single_pass_select(
    question: str,
    serialized: str,
    seed_ids: tuple[UUID, ...],
    cfg: RetrievalCfg,
) -> _RetrievalSelectionDraft:
    agent = Agent(
        name="structure_single_pass_retriever",
        instructions=_SINGLE_PASS_INSTRUCTIONS,
        model=cfg.model,
        model_settings=cfg.model_settings or ModelSettings(),
        output_type=_RetrievalSelectionDraft,
    )
    input_value = json.dumps(
        {
            "question": question,
            "seed_node_ids": [str(value) for value in seed_ids],
            "structure": json.loads(serialized),
        },
        ensure_ascii=False,
    )
    try:
        result = await asyncio.wait_for(
            Runner.run(
                agent,
                input_value,
                run_config=_run_config(cfg),
                max_turns=1,
            ),
            timeout=cfg.timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise RetrievalError(
            "single-pass retrieval exceeded timeout_seconds"
        ) from exc
    return _RetrievalSelectionDraft.model_validate(result.final_output)


async def _agentic_select(
    structure: StructureTree,
    identity: _ResolvableStructure,
    question: str,
    serialized: str,
    seed_ids: tuple[UUID, ...],
    library: LocalKnowledgeLibrary,
    cfg: RetrievalCfg,
) -> _RetrievalSelectionDraft:
    @function_tool
    async def get_document_structure() -> str:
        """Return the bounded document hierarchy without source text."""
        return serialized

    @function_tool
    async def get_node_content(node_ids: list[str]) -> str:
        """Resolve page-cited source content for structure node UUIDs.

        Args:
            node_ids: Canonical node UUID strings from the document structure.
        """
        if len(node_ids) > cfg.max_nodes_per_tool_call:
            raise ValueError("get_node_content exceeds max_nodes_per_tool_call")
        values = []
        for value in node_ids:
            try:
                node_id = UUID(value)
            except ValueError as exc:
                raise ValueError(
                    "get_node_content received an invalid UUID"
                ) from exc
            if node_id not in structure.nodes:
                raise ValueError("get_node_content received an unknown node ID")
            evidence = await _resolve_evidence(
                structure,
                identity,
                node_id,
                library,
            )
            values.append(evidence.model_dump(mode="json"))
        return json.dumps(values, ensure_ascii=False)

    agent = Agent(
        name="structure_agentic_retriever",
        instructions=_AGENTIC_INSTRUCTIONS,
        model=cfg.model,
        model_settings=cfg.model_settings or ModelSettings(),
        tools=[get_document_structure, get_node_content],
        output_type=_RetrievalSelectionDraft,
    )
    input_value = json.dumps(
        {
            "question": question,
            "seed_node_ids": [str(value) for value in seed_ids],
        },
        ensure_ascii=False,
    )
    try:
        result = await asyncio.wait_for(
            Runner.run(
                agent,
                input_value,
                run_config=_run_config(cfg),
                max_turns=cfg.max_turns,
            ),
            timeout=cfg.timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise RetrievalError(
            "agentic retrieval exceeded timeout_seconds"
        ) from exc
    return _RetrievalSelectionDraft.model_validate(result.final_output)


def _validate_selection(
    structure: StructureTree,
    selection: _RetrievalSelectionDraft,
    cfg: RetrievalCfg,
) -> tuple[UUID, ...]:
    selected: list[UUID] = []
    for node_id in selection.node_ids:
        if node_id not in structure.nodes:
            raise ValueError("retrieval selected an unknown structure node")
        if node_id not in selected:
            selected.append(node_id)
    if len(selected) > cfg.max_evidence_nodes:
        raise RetrievalError("retrieval exceeds max_evidence_nodes")
    return tuple(selected)


async def _resolve_evidence(
    structure: StructureTree,
    identity: _ResolvableStructure,
    node_id: UUID,
    library: LocalKnowledgeLibrary,
) -> RetrievalEvidence:
    locator = ArtifactLocator(
        source_revision_id=identity.source_revision_id,
        artifact_id=identity.id,
        artifact_kind=identity.artifact_kind,
        member_id=node_id,
    )
    resolved = await library.resolve(locator)
    if not isinstance(resolved, TreeNode) or not resolved.content:
        raise RetrievalError(
            "library did not resolve structure node source content"
        )
    return RetrievalEvidence(
        locator=locator,
        title=structure.nodes[node_id].title,
        content=resolved.content,
        citations=tuple(resolved.citations),
    )
