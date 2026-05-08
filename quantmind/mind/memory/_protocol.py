"""Memory Protocol — granular cross-step working memory contract.

Each method has its own narrow surface so concrete implementations can
opt in to whichever channel(s) they need (in-process tools, MCP servers,
lifecycle hooks). The Protocol does NOT prescribe MCP: a future
embedding-based ``ChromaMemory`` could be tool-only, while the MVP
``FilesystemMemory`` is MCP-based — both satisfy this same Protocol.
"""

from typing import Any, Protocol, runtime_checkable

from agents import RunHooks, Tool
from agents.mcp import MCPServer


@runtime_checkable
class Memory(Protocol):
    """Cross-step working memory exposed to a flow's Agent."""

    def tools(self) -> list[Tool]:
        """In-process ``@function_tool`` list for the Agent."""
        ...

    def mcp_servers(self) -> list[MCPServer]:
        """MCP servers exposed to the Agent."""
        ...

    def run_hooks(self) -> RunHooks[Any] | None:
        """Lifecycle hooks for accounting / archiving / item indexing."""
        ...

    async def reset(self) -> None:
        """Wipe the memory area. Implementations may be no-ops."""
        ...
