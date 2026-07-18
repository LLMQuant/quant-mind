"""FlattenKnowledge — atomic-card shape.

A flatten card is semantically indivisible: one source artifact maps to one
card whose body is the answer (no structure to navigate). Flatten is the
right shape for `News`, `Earnings`, `Factor`, and `Thesis`. Paper source
revisions and artifacts use their dedicated source-first models.

This file only declares the marker base; concrete subclasses live in
`paper.py` / `news.py` / `earnings.py` / etc.
"""

from quantmind.knowledge._base import BaseKnowledge


class FlattenKnowledge(BaseKnowledge):
    """Marker base for flat domain cards.

    Subclasses add a typed payload (e.g. ``headline``, ``guidance``, or
    ``revenue``). Search text is selected by the library projection boundary.
    """
