"""Earnings knowledge schema (output of earnings_flow)."""

from typing import Literal

from pydantic import Field

from quantmind.knowledge._flatten import FlattenKnowledge


class Earnings(FlattenKnowledge):
    """An earnings-release extraction (one filing -> one Earnings card)."""

    item_type: Literal["earnings"] = "earnings"

    ticker: str
    period: str  # e.g. "2026Q1", "FY2025"
    revenue: float | None = None
    eps: float | None = None
    guidance: str | None = None
    surprise_flags: list[str] = Field(default_factory=list)
    transcript_quote: str | None = None

    def embedding_text(self) -> str:
        guidance = self.guidance or ""
        return f"{self.ticker} {self.period} earnings\n{guidance}".strip()
