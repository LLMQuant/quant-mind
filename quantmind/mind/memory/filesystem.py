"""``FilesystemMemory`` — MVP cross-step memory backed by an MCP filesystem server.

Layout::

    <memory_dir>/
        notes/                # agent's free-form working notes (markdown)
        items/                # typed KnowledgeItem JSON (PR7+)
        runs/                 # trajectory archive — runs/<run_id>.json
        runs.jsonl            # append-only run index
        README.md             # agent-facing usage guide

Requires Node.js + ``npx`` on PATH (the SDK's ``MCPServerStdio`` spawns
``npx -y @modelcontextprotocol/server-filesystem``). ``__init__`` does
**not** pre-flight-check ``npx``; the SDK surfaces a clear error at run
time when it is missing.
"""

import asyncio
import shutil
from pathlib import Path
from typing import Any

from agents import RunHooks, Tool
from agents.mcp import MCPServer, MCPServerStdio

from quantmind.mind.memory._run_hooks import MemoryRunHooks

_AGENT_README_TEXT = """\
# Memory directory for QuantMind flow run

Available subdirectories:
- `notes/` — your free-form working notes (markdown). Read existing notes
  before writing new ones.
- `items/` — structured KnowledgeItem JSON files emitted by previous runs.
- `runs/` — system-managed run trajectory logs (do not edit).
- `runs.jsonl` — append-only run index (system-managed, do not edit).

Guidelines:
1. BEFORE doing your task, list `notes/` and `items/` to see relevant
   prior context.
2. When you find a fact worth remembering, write a short markdown note
   under `notes/`, named `<topic>_<short_slug>.md`.
3. Don't repeat work. If the same item is already in `items/`, prefer
   to reference rather than re-extract.
"""


def _is_forbidden_path(path: Path) -> bool:
    return path == Path("/") or path == Path.home()


class FilesystemMemory:
    """Filesystem-backed cross-step memory using the MCP filesystem server.

    Constructed once per serial loop / batch; passed to ``paper_flow``
    via the ``memory=`` kwarg. Each ``run_hooks()`` invocation returns
    a fresh ``MemoryRunHooks`` so per-run accumulator state is isolated;
    they share the per-instance ``asyncio.Lock`` that serialises
    ``runs.jsonl`` appends.
    """

    def __init__(self, memory_dir: str | Path) -> None:
        self.memory_dir = Path(memory_dir).resolve()
        if _is_forbidden_path(self.memory_dir):
            raise ValueError(
                "memory_dir must not be '/' or the user home directory; "
                "choose a dedicated subdirectory."
            )
        for sub in ("notes", "items", "runs"):
            (self.memory_dir / sub).mkdir(parents=True, exist_ok=True)
        readme = self.memory_dir / "README.md"
        if not readme.exists():
            readme.write_text(_AGENT_README_TEXT, encoding="utf-8")
        self._archive_lock = asyncio.Lock()

    def tools(self) -> list[Tool]:
        return []

    def mcp_servers(self) -> list[MCPServer]:
        return [
            MCPServerStdio(
                name="quantmind_memory_fs",
                params={
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(self.memory_dir),
                    ],
                },
            )
        ]

    def run_hooks(self) -> RunHooks[Any] | None:
        return MemoryRunHooks(
            memory_dir=self.memory_dir,
            archive_lock=self._archive_lock,
        )

    async def reset(self) -> None:
        """Wipe ``notes/``, ``items/``, ``runs/``, and ``runs.jsonl``.

        Destructive — irreversibly removes every file under those paths.
        Re-creates the empty subdirectories and seeds the agent README
        afterwards.
        """
        for sub in ("notes", "items", "runs"):
            shutil.rmtree(self.memory_dir / sub, ignore_errors=True)
        runs_jsonl = self.memory_dir / "runs.jsonl"
        if runs_jsonl.exists():
            runs_jsonl.unlink()
        for sub in ("notes", "items", "runs"):
            (self.memory_dir / sub).mkdir(parents=True, exist_ok=True)
        readme = self.memory_dir / "README.md"
        if not readme.exists():
            readme.write_text(_AGENT_README_TEXT, encoding="utf-8")
