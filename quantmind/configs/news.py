"""News-flow configuration + input discriminated union."""

from typing import Annotated, Literal, Union

from pydantic import Field

from quantmind.configs.base import BaseFlowCfg, BaseInput


class RssFeed(BaseInput):
    """RSS/Atom feed URL to be polled for items."""

    type: Literal["rss"] = "rss"
    url: str


class HttpUrl(BaseInput):
    """Single news article URL."""

    type: Literal["http"] = "http"
    url: str


class Headline(BaseInput):
    """Inline headline text (no body fetching)."""

    type: Literal["headline"] = "headline"
    text: str


NewsInput = Annotated[
    Union[RssFeed, HttpUrl, Headline],
    Field(discriminator="type"),
]


class NewsFlowCfg(BaseFlowCfg):
    """Knobs specific to news_flow."""

    materiality_threshold: Literal["low", "medium", "high"] = "medium"
    entities_hint: list[str] = Field(default_factory=list)
