"""FlattenKnowledge — atomic-card shape.

A flatten card is semantically indivisible: one source artifact maps to one
card whose body is the answer (no structure to navigate). Flatten is the
right shape for `News`, `Earnings`, `Factor`, `Thesis`, and the summary card
of a paper (`PaperKnowledgeCard`). Whole research papers are NOT flatten —
they are `TreeKnowledge`.

This file only declares the marker base; concrete subclasses live in
`paper.py` / `news.py` / `earnings.py` / etc.
"""

from quantmind.knowledge._base import BaseKnowledge


class FlattenKnowledge(BaseKnowledge):
    """Marker base for flat domain cards.

    Subclasses add a typed payload (e.g. ``summary``, ``methodology``,
    ``revenue``). They MUST override `embedding_text()` to produce a stable
    string suitable for the store's vector index.
    """
