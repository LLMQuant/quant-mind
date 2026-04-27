"""Earnings-flow configuration + input discriminated union."""

from typing import Annotated, Literal, Union

from pydantic import Field

from quantmind.configs.base import BaseFlowCfg, BaseInput


class TickerPeriod(BaseInput):
    """Ticker + reporting period (e.g. ``AAPL`` / ``2026Q1``)."""

    type: Literal["ticker_period"] = "ticker_period"
    ticker: str
    period: str  # e.g. "2026Q1"


class TranscriptText(BaseInput):
    """Raw earnings-call transcript pasted inline."""

    type: Literal["transcript"] = "transcript"
    text: str


class HttpUrl(BaseInput):
    """URL to an earnings release / IR filing."""

    type: Literal["http"] = "http"
    url: str


EarningsInput = Annotated[
    Union[TickerPeriod, TranscriptText, HttpUrl],
    Field(discriminator="type"),
]


class EarningsFlowCfg(BaseFlowCfg):
    """Knobs specific to earnings_flow."""

    detect_surprises: bool = True
    include_guidance: bool = True
