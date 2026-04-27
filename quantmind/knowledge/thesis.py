"""Thesis knowledge schema (stub; full payload lands with thesis_flow).

The class exists today so ``from quantmind.knowledge import Thesis`` is
stable across PRs. Concrete thesis fields (assumptions / evidence_refs /
time_horizon) are added when ``thesis_flow`` ships.
"""

from typing import Literal

from quantmind.knowledge._flatten import FlattenKnowledge


class Thesis(FlattenKnowledge):
    """An investment thesis card (stub)."""

    item_type: Literal["thesis"] = "thesis"

    claim: str

    def embedding_text(self) -> str:
        return self.claim
