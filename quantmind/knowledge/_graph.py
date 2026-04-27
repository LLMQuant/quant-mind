"""GraphKnowledge — cross-item edges (placeholder, NOT implemented).

The graph shape covers relations BETWEEN knowledge items: paper-cites-paper,
factor-derived-from-factor, news-mentions-ticker. It is distinct from the
internal hierarchy of a single item (which is `TreeKnowledge`).

This module exists so users can write::

    Knowledge = FlattenKnowledge | TreeKnowledge | GraphKnowledge

honestly today, even though the implementation is deferred. Subclassing is
blocked at class-creation time so no one accidentally builds against an
unfinalised contract.
"""

from typing import Any

from quantmind.knowledge._base import BaseKnowledge


class GraphKnowledge(BaseKnowledge):
    """Reserved for cross-item edges. NOT IMPLEMENTED.

    Future shape (sketched, subject to change):

    - ``nodes: dict[UUID, NodeRef]`` where ``NodeRef = (knowledge_id, knowledge_type)``
    - ``edges: list[Edge]`` where ``Edge = (from, to, kind, weight, evidence)``

    Use cases under consideration: paper citation graph, factor lineage,
    news–entity co-occurrence. An issue will be opened when the first
    concrete need arrives (PR8+).
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        raise NotImplementedError(
            "GraphKnowledge is a design-intent placeholder; subclassing is "
            "blocked until the shape is finalised. Open a tracking issue "
            "before lifting this guard."
        )
