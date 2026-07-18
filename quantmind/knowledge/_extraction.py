"""Slug -> UUID canonicalisation for LLM-extracted TreeKnowledge.

With ``strict_json_schema=False`` the openai-agents SDK lets the model fill
``UUID`` id fields with human-readable slugs ("root", "intro"). Slugs are the
right shape for one-shot tree generation: the model references nodes it is
creating in the same response, and self-consistent slugs beat invented UUIDs.
The domain model still stores ``UUID``: a deliberate, tested invariant (node-id
uniqueness + UUID-typed JSON round-trips in ``tests/knowledge``) and the basis
for stable dedup/identity across re-runs. This module bridges the
two: it maps each distinct slug to a fresh ``UUID`` and rewrites every id slot,
leaving values that are already valid UUIDs untouched.
"""

from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

_ResolveFn = Callable[[Any], Any]


def _looks_like_uuid(value: Any) -> bool:
    """True if ``value`` is a UUID or a string already in UUID form."""
    if isinstance(value, UUID):
        return True
    if not isinstance(value, str):
        return False
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def canonicalize_tree_ids(data: Any) -> Any:
    """Rewrite slug ids in a TreeKnowledge-shaped mapping to UUIDs.

    Pure and copy-on-write: the input is never mutated. A no-op for anything
    that is not a ``dict`` carrying a ``nodes`` map, so non-tree payloads and
    already-canonical trees pass straight through. Every distinct slug maps to
    one UUID, so cross-references (``parent_id``, ``children_ids``, citation
    anchors, and ``nodes`` keys) stay internally consistent.
    """
    if not isinstance(data, dict):
        return data
    nodes = data.get("nodes")
    if not isinstance(nodes, dict):
        return data

    mapping: dict[str, str] = {}

    def resolve(raw: Any) -> Any:
        """Map one id slot: slug -> uuid; UUID / None / non-str pass through."""
        if raw is None or not isinstance(raw, str) or _looks_like_uuid(raw):
            return raw
        if raw not in mapping:
            mapping[raw] = str(uuid4())
        return mapping[raw]

    # Seed from the authoritative identity slots (the node keys) first so a
    # reference resolves to the same UUID no matter where it is first seen.
    for slug in nodes:
        resolve(slug)

    out = dict(data)
    if "id" in out:
        out["id"] = resolve(out["id"])
    if "root_node_id" in out:
        out["root_node_id"] = resolve(out["root_node_id"])
    out["citations"] = [
        _rewrite_citation(c, resolve) for c in out.get("citations", [])
    ]
    out["nodes"] = {
        resolve(key): _rewrite_node(node, resolve)
        for key, node in nodes.items()
    }
    return out


def _rewrite_node(node: Any, resolve: _ResolveFn) -> Any:
    """Return a copy of one node dict with its id slots resolved."""
    if not isinstance(node, dict):
        return node
    out = dict(node)
    if "node_id" in out:
        out["node_id"] = resolve(out["node_id"])
    if "parent_id" in out:
        out["parent_id"] = resolve(out["parent_id"])
    children = out.get("children_ids")
    if isinstance(children, list):
        out["children_ids"] = [resolve(c) for c in children]
    citations = out.get("citations")
    if isinstance(citations, list):
        out["citations"] = [_rewrite_citation(c, resolve) for c in citations]
    return out


def _rewrite_citation(cit: Any, resolve: _ResolveFn) -> Any:
    """Return a copy of one citation dict with its anchor ids resolved."""
    if not isinstance(cit, dict):
        return cit
    out = dict(cit)
    if "tree_id" in out:
        out["tree_id"] = resolve(out["tree_id"])
    if "node_id" in out:
        out["node_id"] = resolve(out["node_id"])
    return out
