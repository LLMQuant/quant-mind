"""Configuration for reasoning-based structure-tree retrieval.

``RetrievalCfg`` is the strategy-agnostic base: it holds the model and the shared
structure/evidence bounds. The concrete strategy is selected by the cfg *type*
handed to ``quantmind.mind.Retrieve`` — this is typed dispatch, not a field on a
single class and not a class hierarchy of retrievers. ``AgenticRetrievalCfg`` is
the one implemented strategy today (agentic traversal over a structure tree);
semantic / hybrid cfg types are reserved seams.
"""

from typing import Literal

from pydantic import Field

from quantmind.configs.base import BaseFlowCfg


class RetrievalCfg(BaseFlowCfg):
    """Shared model and bounds for every structure-tree retrieval strategy.

    A bare ``RetrievalCfg`` names no implemented strategy; pass a concrete
    subclass (``AgenticRetrievalCfg``) to select one. The SDK / runtime fields
    (``model_settings``, ``max_turns``, ``timeout_seconds``, ``workflow_name``,
    the ``trace_*`` knobs, ``tracing_disabled``) are inherited from
    ``BaseFlowCfg``.
    """

    model: str = "gpt-4o-mini"
    structure_token_budget: int = Field(default=8_000, ge=256)
    max_evidence_nodes: int = Field(default=8, ge=1)


class AgenticRetrievalCfg(RetrievalCfg):
    """Agentic traversal over a self-contained structure tree.

    ``mode`` names the structure kind traversed (``"tree"`` today; ``"graph"``
    is the reserved future value). ``extra_instruction`` is appended verbatim to
    the agent's traversal instructions. ``max_nodes_per_tool_call`` bounds a
    single ``get_node_content`` request; the traversal-loop bound ``max_turns``
    is inherited from ``BaseFlowCfg``.
    """

    mode: Literal["tree"] = "tree"
    extra_instruction: str | None = None
    max_nodes_per_tool_call: int = Field(default=8, ge=1)
