"""Shared tree structure and conventional hierarchical knowledge shape.

A tree's structure carries information: nodes derive meaning from their
position under a parent. `TreeKnowledge` is the right shape for regulatory
filings (10-K parts), earnings-call transcripts (intro / Q&A / per-question),
and a future paper-navigation artifact.

Retrieval over a tree is reasoning-driven (PageIndex-style): an agent reads
the root summary plus children summaries, picks the most likely branch,
drills down, and lazy-loads leaf content. Embeddings (when available) act as
a coarse pre-filter, never as a replacement for that reasoning.
"""

from collections.abc import Collection, Iterator
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from quantmind.knowledge._base import BaseKnowledge, Citation


class StructureTreeValidationError(ValueError):
    """A structure tree failed its code-owned integrity gate."""


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


class StructureTree(BaseModel):
    """Identity-free tree payload shared by document-specific bindings.

    Holds the full set of nodes in a flat ``nodes`` dict for O(1) lookup,
    plus the ``root_node_id`` pointer. Subclasses add canonical or derived
    artifact identity without nesting a second tree model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

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
        seen: set[UUID] = set()
        cursor: UUID | None = node_id
        while cursor is not None:
            if cursor in seen or cursor not in self.nodes:
                return []
            seen.add(cursor)
            node = self.nodes[cursor]
            path.append(node)
            cursor = node.parent_id
        path.reverse()
        return path if path[0].node_id == self.root_node_id else []

    def validate(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        source_pages: Collection[int] | None = None,
    ) -> None:
        """Reject malformed topology, links, positions, and page ownership.

        Args:
            source_pages: Optional physical pages owned by the exact source.
                Document-specific bindings pass this value so citation bounds
                can be checked without putting source identity on the base.

        Raises:
            StructureTreeValidationError: If any integrity rule fails.
        """
        if not self.nodes:
            raise StructureTreeValidationError("structure tree has no nodes")
        if self.root_node_id not in self.nodes:
            raise StructureTreeValidationError(
                "structure tree root_node_id is not present in nodes"
            )
        for node_id, node in self.nodes.items():
            if node_id != node.node_id:
                raise StructureTreeValidationError(
                    "structure tree node map key does not match node_id"
                )

        roots = [node for node in self.nodes.values() if node.parent_id is None]
        if len(roots) != 1 or roots[0].node_id != self.root_node_id:
            raise StructureTreeValidationError(
                "structure tree must have exactly one declared root"
            )

        sibling_positions: dict[UUID | None, set[int]] = {}
        for node in self.nodes.values():
            positions = sibling_positions.setdefault(node.parent_id, set())
            if node.position in positions:
                raise StructureTreeValidationError(
                    "structure tree siblings have duplicate positions"
                )
            positions.add(node.position)
            if node.parent_id is not None:
                parent = self.nodes.get(node.parent_id)
                if parent is None:
                    raise StructureTreeValidationError(
                        "structure tree contains an orphan node"
                    )
                if node.node_id not in parent.children_ids:
                    raise StructureTreeValidationError(
                        "structure tree parent/children links are inconsistent"
                    )
            if len(set(node.children_ids)) != len(node.children_ids):
                raise StructureTreeValidationError(
                    "structure tree contains duplicate child links"
                )
            for child_id in node.children_ids:
                child = self.nodes.get(child_id)
                if child is None or child.parent_id != node.node_id:
                    raise StructureTreeValidationError(
                        "structure tree parent/children links are inconsistent"
                    )

        for start_id in self.nodes:
            chain: set[UUID] = set()
            cursor: UUID | None = start_id
            while cursor is not None:
                if cursor in chain:
                    raise StructureTreeValidationError(
                        "structure tree contains a cycle"
                    )
                chain.add(cursor)
                cursor = self.nodes[cursor].parent_id

        allowed_pages = set(source_pages) if source_pages is not None else None
        page_sets: dict[UUID, set[int]] = {}
        for node in self.nodes.values():
            pages: set[int] = set()
            for citation in node.citations:
                if citation.page is None or citation.page < 1:
                    raise StructureTreeValidationError(
                        "structure tree citations require a positive page"
                    )
                if (
                    allowed_pages is not None
                    and citation.page not in allowed_pages
                ):
                    raise StructureTreeValidationError(
                        "structure tree citation page is outside its source"
                    )
                pages.add(citation.page)
            page_sets[node.node_id] = pages

        visited: set[UUID] = set()
        active: set[UUID] = set()

        def visit(node_id: UUID) -> None:
            if node_id in active:
                raise StructureTreeValidationError(
                    "structure tree contains a cycle"
                )
            if node_id in visited:
                return
            active.add(node_id)
            node = self.nodes[node_id]
            for child_id in node.children_ids:
                if not page_sets[child_id].issubset(page_sets[node_id]):
                    raise StructureTreeValidationError(
                        "structure tree child pages are not contained in parent"
                    )
                visit(child_id)
            active.remove(node_id)
            visited.add(node_id)

        visit(self.root_node_id)
        if visited != set(self.nodes):
            raise StructureTreeValidationError(
                "structure tree contains an unreachable node"
            )


class TreeKnowledge(BaseKnowledge, StructureTree):
    """Conventional hierarchical knowledge using the shared tree payload."""
