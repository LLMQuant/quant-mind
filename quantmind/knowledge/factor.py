"""Factor knowledge schema (stub; full payload lands with factor_flow).

The class exists today so ``from quantmind.knowledge import Factor`` is
stable across PRs. Concrete factor research fields (universe / period / pnl
/ ic / turnover) are added when ``factor_flow`` ships.
"""

from typing import Literal

from quantmind.knowledge._flatten import FlattenKnowledge


class Factor(FlattenKnowledge):
    """A single factor card (stub)."""

    item_type: Literal["factor"] = "factor"

    factor_name: str
    universe: str | None = None

    def embedding_text(self) -> str:
        scope = self.universe or "unspecified"
        return f"factor {self.factor_name} on {scope}"
