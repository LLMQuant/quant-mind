"""Paper-flow configuration + input discriminated union.

`PaperInput` is one of:
- `ArxivIdentifier`: arxiv id or full URL parsed by preprocess.fetch.arxiv
- `HttpUrl`: any web URL (PDF or HTML; routed by content-type)
- `LocalFilePath`: filesystem path to a PDF / HTML / Markdown file
- `RawText`: an inline string (for tests or LLM-pre-cleaned inputs)
- `DoiIdentifier`: a DOI to be resolved via preprocess.fetch.doi
"""

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import Field

from quantmind.configs.base import BaseFlowCfg, BaseInput


class ArxivIdentifier(BaseInput):
    """Arxiv id (e.g. ``2604.12345``) or full arxiv URL."""

    type: Literal["arxiv"] = "arxiv"
    id: str


class HttpUrl(BaseInput):
    """Any web URL; PDF vs HTML is decided by content-type."""

    type: Literal["http"] = "http"
    url: str


class LocalFilePath(BaseInput):
    """Filesystem path to a PDF / HTML / Markdown file."""

    type: Literal["local"] = "local"
    path: Path


class RawText(BaseInput):
    """Inline text input (tests / pre-cleaned content)."""

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
    """Knobs specific to paper_flow."""

    extract_methodology: bool = True
    extract_limitations: bool = True
    asset_class_hint: str | None = None
