"""Configuration for source-native paper structure construction."""

from pydantic import Field

from quantmind.configs.base import BaseFlowCfg


class PaperStructureCfg(BaseFlowCfg):
    """Model, prompt, input, and tree bounds for ``PaperFlow(cfg).build``.

    ``PaperFlow.build`` dispatches on the cfg **type**: constructing
    ``PaperFlow`` with a ``PaperStructureCfg`` selects the self-contained
    ``PaperStructureTree`` shape.
    """

    model: str = "gpt-4o-mini"
    prompt_version: str = "paper-structure-v2"
    instructions: str | None = None
    page_text_chars: int = Field(default=1_200, ge=80)
    max_output_tokens: int = Field(default=4_096, gt=0)
    max_depth: int = Field(default=6, ge=1)
    max_nodes: int = Field(default=128, ge=1)
