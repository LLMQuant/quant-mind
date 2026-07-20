"""Vectorless reasoning-based retrieval over a self-contained structure tree.

Single-tree retrieval is a pure value operation: the tree carries its own node
content, so ``retrieve`` reads evidence text straight from ``tree.nodes`` and
returns evidence *values*. It binds no library and imports none. The reference
(``ArtifactLocator``) rides along only as optional provenance for future
cross-artifact fusion, never as the path a consumer must resolve to see content.
"""

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
)

_SINGLE_PASS_INSTRUCTIONS = """\
Select the structure-tree nodes that contain evidence relevant to the question.
Reason only over the supplied titles, summaries, hierarchy, and optional seed
nodes. Prefer leaf nodes, which carry the actual source text; an internal node
stands in only for the union of its leaves. Return node UUIDs, not an answer.
Prefer the smallest sufficient set and never invent an ID.
"""

_AGENTIC_INSTRUCTIONS = """\
Retrieve evidence relevant to the question by inspecting the document structure
and opening node content as needed. Prefer leaf nodes, which carry the actual
source text; opening an internal node returns the concatenation of its leaves.
Return the UUIDs of the smallest sufficient evidence nodes, not an answer. Never
invent an ID. Seed nodes are hints only; you may inspect other branches when the
structure supports it.
"""


class RetrievalError(RuntimeError):
    """Reasoning-based retrieval exceeded a runtime or evidence boundary."""


class RetrievalEvidence(BaseModel):
    """One selected node's self-contained, page-cited source content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    content: str = Field(min_length=1)
    citations: tuple[Citation, ...] = ()
    locator: ArtifactLocator | None = None


class _RetrievalSelectionDraft(BaseModel):
    """Model-selected canonical node coordinates before code resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_ids: tuple[UUID, ...] = Field(min_length=1)


@runtime_checkable
class _ResolvableStructure(Protocol):
    """Identity a tree exposes when it can address itself as an artifact."""

    id: UUID
    source_revision_id: UUID
    artifact_kind: str


async def retrieve(
    tree: StructureTree,
    question: str,
    *,
    cfg: RetrievalCfg | None = None,
    seed_node_ids: list[UUID] | None = None,
) -> list[RetrievalEvidence]:
    """Select and read page-cited evidence from one self-contained tree.

    This is a library-free pure value operation: node content is read from the
    tree itself (see ``_node_text``), so every returned evidence value carries
    its own content and needs no downstream resolution. When the tree is
    identity-bearing (e.g. a ``PaperStructureTree``), each evidence value also
    carries an ``ArtifactLocator`` for optional cross-artifact fusion; otherwise
    ``locator`` is ``None`` and content is still present.

    Dispatch is on ``cfg.grain`` today:

    - ``single-pass``: serialize the tree (ids, titles, summaries, hierarchy —
      leaf text stripped for the token budget), make one model call for the
      relevant node ids, then read those nodes' content from the tree.
    - ``agentic``: expose ``get_document_structure()`` and
      ``get_node_content(node_ids)`` tools that read the in-memory tree, and let
      an Agent traverse turn by turn.

    Future seam: a widened ``retrieve`` is intended to also dispatch to
    ``semantic`` search over a vector-backed knowledge shape and to ``hybrid``
    (semantic shortlist, then agentic reasoning over it) when both are
    available. Those grains are deliberately not implemented here; a single
    function that branches on cfg/knowledge kind is internal dispatch, not the
    forbidden generic retriever/query-engine hierarchy.

    Args:
        tree: Validated self-contained structure tree whose leaf nodes carry
            their own source content.
        question: Evidence need used for relevance reasoning.
        cfg: Model, retrieval grain, and structure/evidence bounds. A default
            ``RetrievalCfg`` is used when omitted.
        seed_node_ids: Optional in-tree node hints (validated against
            ``tree.nodes``) reserved for a later semantic shortlist. This
            operation performs no semantic search and takes no library seeds.

    Returns:
        Selected node evidence with self-contained content and optional
        provenance.

    Raises:
        ValueError: If the question, seeds, or selected node IDs are invalid.
        RetrievalError: If runtime or evidence bounds are exceeded, or a
            selected node yields no content.
    """
    active_cfg = (cfg or RetrievalCfg()).model_copy(deep=True)
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("retrieval question must not be blank")
    tree.validate()
    seed_ids = _validate_seed_node_ids(tree, seed_node_ids or [])
    serialized = _serialize_structure(
        tree,
        token_budget=active_cfg.structure_token_budget,
        seed_ids=set(seed_ids),
    )
    if active_cfg.grain == "single-pass":
        selection = await _single_pass_select(
            normalized_question,
            serialized,
            seed_ids,
            active_cfg,
        )
    else:
        selection = await _agentic_select(
            tree,
            normalized_question,
            serialized,
            seed_ids,
            active_cfg,
        )
    node_ids = _validate_selection(tree, selection, active_cfg)
    return [_node_evidence(tree, node_id) for node_id in node_ids]


def _validate_seed_node_ids(
    tree: StructureTree,
    seed_node_ids: list[UUID],
) -> tuple[UUID, ...]:
    node_ids: list[UUID] = []
    for node_id in seed_node_ids:
        if node_id not in tree.nodes:
            raise ValueError("seed_node_ids must address nodes in the tree")
        if node_id not in node_ids:
            node_ids.append(node_id)
    return tuple(node_ids)


def _node_text(tree: StructureTree, node_id: UUID) -> str:
    """Return a node's self-contained text.

    A leaf yields its own ``content``; an internal node yields the reading-order
    concatenation of its descendant leaves' content, so evidence is non-empty
    even when the model selects an internal node.
    """
    node = tree.nodes[node_id]
    if not node.children_ids:
        return node.content or ""
    texts: list[str] = []

    def collect(current_id: UUID) -> None:
        current = tree.nodes[current_id]
        if not current.children_ids:
            if current.content:
                texts.append(current.content)
            return
        for child_id in current.children_ids:
            collect(child_id)

    collect(node_id)
    return "\n\n".join(texts)


def _tree_locator(tree: StructureTree, node_id: UUID) -> ArtifactLocator | None:
    if not isinstance(tree, _ResolvableStructure):
        return None
    return ArtifactLocator(
        source_revision_id=tree.source_revision_id,
        artifact_id=tree.id,
        artifact_kind=tree.artifact_kind,
        member_id=node_id,
    )


def _node_evidence(tree: StructureTree, node_id: UUID) -> RetrievalEvidence:
    node = tree.nodes[node_id]
    content = _node_text(tree, node_id)
    if not content:
        raise RetrievalError("structure node yielded no resolvable content")
    return RetrievalEvidence(
        title=node.title,
        content=content,
        citations=tuple(node.citations),
        locator=_tree_locator(tree, node_id),
    )


def _serialize_structure(
    tree: StructureTree,
    *,
    token_budget: int,
    seed_ids: set[UUID],
) -> str:
    character_budget = token_budget * 4
    payload: dict[str, object] = {
        "root_node_id": str(tree.root_node_id),
        "nodes": [],
        "truncated": False,
    }
    serialized_nodes = cast(list[dict[str, object]], payload["nodes"])
    for node in tree.walk_dfs():
        candidate: dict[str, object] = {
            "node_id": str(node.node_id),
            "parent_id": str(node.parent_id) if node.parent_id else None,
            "position": node.position,
            "title": node.title,
            "summary": node.summary,
            "is_leaf": not node.children_ids,
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
    tree: StructureTree,
    question: str,
    serialized: str,
    seed_ids: tuple[UUID, ...],
    cfg: RetrievalCfg,
) -> _RetrievalSelectionDraft:
    @function_tool
    async def get_document_structure() -> str:
        """Return the bounded document hierarchy without source text."""
        return serialized

    @function_tool
    async def get_node_content(node_ids: list[str]) -> str:
        """Read self-contained source content for structure node UUIDs.

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
            if node_id not in tree.nodes:
                raise ValueError("get_node_content received an unknown node ID")
            values.append(_node_evidence(tree, node_id).model_dump(mode="json"))
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
    tree: StructureTree,
    selection: _RetrievalSelectionDraft,
    cfg: RetrievalCfg,
) -> tuple[UUID, ...]:
    selected: list[UUID] = []
    for node_id in selection.node_ids:
        if node_id not in tree.nodes:
            raise ValueError("retrieval selected an unknown structure node")
        if node_id not in selected:
            selected.append(node_id)
    if len(selected) > cfg.max_evidence_nodes:
        raise RetrievalError("retrieval exceeds max_evidence_nodes")
    return tuple(selected)
