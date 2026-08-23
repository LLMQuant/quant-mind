"""Configuration for source-native paper structure construction."""

from pydantic import Field

from quantmind.configs.base import BaseFlowCfg


class PaperStructureCfg(BaseFlowCfg):
    """Model, prompt, input, and tree bounds for ``PaperFlow(cfg).build``.

    ``PaperFlow.build`` dispatches on the cfg **type**: constructing
    ``PaperFlow`` with a ``PaperStructureCfg`` selects the self-contained
    ``PaperStructureTree`` shape.

    Drafting reads full page text in character-bounded windows
    (``window_chars`` per model call, ``window_overlap_pages`` shared pages
    between consecutive windows). ``page_text_chars`` is an optional per-page
    clip for cost control on sparse inputs; the ``None`` default sends each
    page complete.
    """

    model: str = "gpt-5.6-luna"
    prompt_version: str = "paper-structure-v3"
    instructions: str | None = None
    page_text_chars: int | None = Field(default=None, ge=80)
    window_chars: int = Field(default=80_000, ge=2_000)
    window_overlap_pages: int = Field(default=1, ge=0)
    max_output_tokens: int = Field(default=4_096, gt=0)
    max_depth: int = Field(default=6, ge=1)
    max_nodes: int = Field(default=128, ge=1)
