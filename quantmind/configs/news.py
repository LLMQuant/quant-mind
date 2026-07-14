"""Typed input and configuration for deterministic news collection."""

from datetime import timezone
from typing import Literal

from pydantic import AwareDatetime, field_validator, model_validator
from typing_extensions import Self

from quantmind.configs.base import BaseFlowCfg, BaseInput


class NewsWindow(BaseInput):
    """A replayable source window using the half-open interval [start, end)."""

    type: Literal["window"] = "window"
    source: Literal["pr-newswire"]
    start: AwareDatetime
    end: AwareDatetime

    @field_validator("start", "end")
    @classmethod
    def normalize_to_utc(cls, value: AwareDatetime) -> AwareDatetime:
        """Normalize aware timestamps so collectors receive one timezone."""
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """Reject empty or reversed collection windows."""
        if self.end <= self.start:
            raise ValueError("NewsWindow requires end to be after start")
        return self


class NewsCollectionCfg(BaseFlowCfg):
    """Collection behavior that changes the returned evidence."""

    retain_raw_html: bool = False
