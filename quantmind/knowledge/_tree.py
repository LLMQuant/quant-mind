"""TreeKnowledge — hierarchical-artifact shape.

A tree's structure carries information: nodes derive meaning from their
position under a parent. `TreeKnowledge` is the right shape for whole
research papers (sections / subsections), regulatory filings (10-K parts),
and earnings-call transcripts (intro / Q&A / per-question).

Retrieval over a tree is reasoning-driven (PageIndex-style): an agent reads
the root summary plus children summaries, picks the most likely branch,
drills down, and lazy-loads leaf content. Embeddings (when available) act as
a coarse pre-filter, never as a replacement for that reasoning.
"""

from collections.abc import Iterator
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from quantmind.knowledge._base import BaseKnowledge, Citation


class TreeNode(BaseModel):
    """A single node in a TreeKnowledge.

    `summary` is mandatory because agents navigate by reading it. `content`
    is the optional full-text body (typically populated only on leaves to
    keep the tree small in memory). `children_ids` is an adjacency list; the
    parent `TreeKnowledge` resolves them via its `nodes` map.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: UUID = Field(default_factory=uuid4)
    parent_id: UUID | None = None
    position: int = 0
    title: str
    summary: str
    content: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    children_ids: list[UUID] = Field(default_factory=list)

    def embedding_text(self) -> str:
        """Default: title + summary. Override per domain if needed."""
        return f"{self.title}\n{self.summary}"


class TreeKnowledge(BaseKnowledge):
    """Hierarchical knowledge artifact.

    Holds the full set of nodes in a flat ``nodes`` dict for O(1) lookup,
    plus the ``root_node_id`` pointer. Whether a backend loads all nodes
    eagerly or lazily is its concern; the schema always represents a
    complete tree.
    """

    root_node_id: UUID
    nodes: dict[UUID, TreeNode]

    def root(self) -> TreeNode:
        return self.nodes[self.root_node_id]

    def children_of(self, node_id: UUID) -> list[TreeNode]:
        node = self.nodes[node_id]
        return [self.nodes[c] for c in node.children_ids]

    def walk_dfs(self) -> Iterator[TreeNode]:
        """Depth-first traversal starting at the root."""
        stack: list[UUID] = [self.root_node_id]
        while stack:
            node_id = stack.pop()
            node = self.nodes[node_id]
            yield node
            # Reverse so children are visited in declared order.
            stack.extend(reversed(node.children_ids))

    def find_path(self, node_id: UUID) -> list[TreeNode]:
        """Root-to-node path. Empty if `node_id` is not in the tree."""
        if node_id not in self.nodes:
            return []
        path: list[TreeNode] = []
        cursor: UUID | None = node_id
        while cursor is not None:
            node = self.nodes[cursor]
            path.append(node)
            cursor = node.parent_id
        path.reverse()
        return path

    def embedding_text(self) -> str:
        """Default: root node's embedding text. Override per domain if needed."""
        return self.root().embedding_text()
