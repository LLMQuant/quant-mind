"""Configuration for reasoning-based structure-tree retrieval."""

from typing import Literal

from pydantic import Field

from quantmind.configs.base import BaseFlowCfg


class RetrievalCfg(BaseFlowCfg):
    """Model, traversal grain, and bounds for structure-tree retrieval."""

    model: str = "gpt-4o-mini"
    grain: Literal["single-pass", "agentic"] = "single-pass"
    structure_token_budget: int = Field(default=8_000, ge=256)
    max_evidence_nodes: int = Field(default=8, ge=1)
    max_nodes_per_tool_call: int = Field(default=8, ge=1)
