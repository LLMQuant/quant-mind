"""Configuration for reasoning-based (agentic) structure-tree retrieval.

``RetrievalCfg`` configures ``quantmind.mind.AgenticRetriever`` — the model, the
structure/evidence bounds, and the agent traversal knobs. There is one strategy:
an LLM agent reasons over a knowledge structure. Mechanical retrieval (semantic
vector search / BM25) is not configured here; it lives in ``quantmind.library``
and ``quantmind.rag``.
"""

from pydantic import Field

from quantmind.configs.base import BaseFlowCfg


class RetrievalCfg(BaseFlowCfg):
    """Model, bounds, and agent knobs for agentic structure-tree retrieval.

    The SDK / runtime fields (``model_settings``, ``timeout_seconds``,
    ``workflow_name``, the ``trace_*`` knobs, ``tracing_disabled``) are inherited
    from ``BaseFlowCfg``. ``extra_instruction`` is appended verbatim to the
    agent's traversal instructions; ``max_nodes_per_tool_call`` bounds a single
    ``get_node_content`` request. ``max_turns`` overrides the base default with
    more head-room, because agentic traversal opens nodes over several turns
    before it finalizes a selection; if it is exhausted, ``retrieve`` raises
    ``RetrievalError``. The structure kind (tree today, graph later) is inferred
    from the ``Retrievable`` passed to ``retrieve``, not declared here.
    """

    model: str = "gpt-5.6-luna"
    max_turns: int = Field(default=20, ge=1)
    structure_token_budget: int = Field(default=8_000, ge=256)
    max_evidence_nodes: int = Field(default=8, ge=1)
    max_nodes_per_tool_call: int = Field(default=8, ge=1)
    extra_instruction: str | None = None
