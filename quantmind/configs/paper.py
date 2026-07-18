"""Paper-flow configuration + input discriminated union.

`PaperInput` describes supported and reserved source variants. Paper Flow V1
accepts PDF-backed arXiv, HTTP, and local inputs. It rejects non-PDF HTTP/local
content and raw text, and reserves DOI input until an exact PDF resolver exists.
"""

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from quantmind.configs.base import BaseFlowCfg, BaseInput


class ArxivIdentifier(BaseInput):
    """Arxiv id (e.g. ``2604.12345``) or full arxiv URL."""

    type: Literal["arxiv"] = "arxiv"
    id: str


class HttpUrl(BaseInput):
    """A web URL that must resolve to a PDF in Paper Flow V1."""

    type: Literal["http"] = "http"
    url: str


class LocalFilePath(BaseInput):
    """Filesystem path to a PDF for Paper Flow V1."""

    type: Literal["local"] = "local"
    path: Path


class RawText(BaseInput):
    """Reserved inline text input rejected by page-aware Paper Flow V1."""

    type: Literal["text"] = "text"
    text: str


class DoiIdentifier(BaseInput):
    """A DOI to be resolved by ``preprocess.fetch.doi``."""

    type: Literal["doi"] = "doi"
    doi: str


PaperInput = Annotated[
    Union[ArxivIdentifier, HttpUrl, LocalFilePath, RawText, DoiIdentifier],
    Field(discriminator="type"),
]


class PaperFlowCfg(BaseFlowCfg):
    """Chunking, summarization, and hard-budget controls for ``paper_flow``."""

    model: str = "gpt-4o-mini"
    max_turns: int = Field(default=16, ge=1)
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    summary_prompt_version: str = "paper-summary-v1"
    summary_instructions: str | None = None
    max_summary_tool_calls: int = Field(default=12, ge=1)
    max_summary_concurrency: int = Field(default=2, ge=1)
    max_summary_input_tokens: int = Field(default=120_000, ge=1)
    max_summary_output_tokens: int = Field(default=4_096, ge=1)
    min_summary_citations: int = Field(default=3, ge=1)
    min_summary_pages: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _validate_paper_bounds(self) -> "PaperFlowCfg":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_summary_pages > self.min_summary_citations:
            raise ValueError(
                "min_summary_pages cannot exceed min_summary_citations"
            )
        return self
