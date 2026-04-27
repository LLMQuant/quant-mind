"""quantmind.knowledge — data standard for extracted financial knowledge.

The standard defines three shapes that share `BaseKnowledge`:

- `FlattenKnowledge` — atomic cards (`News`, `Earnings`, `PaperKnowledgeCard`).
- `TreeKnowledge` — hierarchical artifacts (`Paper`).
- `GraphKnowledge` — cross-item edges (placeholder, not implemented).

Every concrete subclass is frozen Pydantic v2 with ``extra="forbid"``,
suitable for ``Agent(output_type=...)`` and round-tripping through JSON.
Subclasses MUST override ``embedding_text()`` so the store layer knows
what to embed.
"""

from quantmind.knowledge._base import (
    BaseKnowledge,
    Citation,
    ExtractionRef,
    SourceRef,
)
from quantmind.knowledge._flatten import FlattenKnowledge
from quantmind.knowledge._graph import GraphKnowledge
from quantmind.knowledge._tree import TreeKnowledge, TreeNode
from quantmind.knowledge.earnings import Earnings
from quantmind.knowledge.factor import Factor
from quantmind.knowledge.news import News
from quantmind.knowledge.paper import Paper, PaperKnowledgeCard
from quantmind.knowledge.thesis import Thesis

__all__ = [
    # Base
    "BaseKnowledge",
    "Citation",
    "ExtractionRef",
    "SourceRef",
    # Shapes
    "FlattenKnowledge",
    "GraphKnowledge",
    "TreeKnowledge",
    "TreeNode",
    # Concrete
    "Earnings",
    "Factor",
    "News",
    "Paper",
    "PaperKnowledgeCard",
    "Thesis",
]
