"""``FilesystemMemory`` — MVP cross-step memory backed by an MCP filesystem server.

Layout::

    <memory_dir>/
        .quantmind-memory     # marker — guards against using a non-QM directory
        workspace/            # MCP root (Agent-visible)
            notes/            # Agent's free-form working notes (markdown)
            items/            # typed KnowledgeItem JSON (PR7+)
            README.md         # Agent-facing usage guide
        runs/                 # system-only trajectory archive (NOT exposed to Agent)
            <run_id>.json
        runs.jsonl            # system-only append-only run index

The MCP filesystem server is rooted at ``workspace/`` so the Agent
cannot read or write the trajectory archive — that keeps run records
tamper-proof under prompt injection.

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
# Memory workspace for QuantMind flow run

Available subdirectories:
- `notes/` — your free-form working notes (markdown). Read existing notes
  before writing new ones.
- `items/` — structured KnowledgeItem JSON files emitted by previous runs.

Guidelines:
1. BEFORE doing your task, list `notes/` and `items/` to see relevant
   prior context.
2. When you find a fact worth remembering, write a short markdown note
   under `notes/`, named `<topic>_<short_slug>.md`.
3. Don't repeat work. If the same item is already in `items/`, prefer
   to reference rather than re-extract.
"""

_MARKER_NAME = ".quantmind-memory"

_FORBIDDEN_PATHS = frozenset(
    {
        Path("/"),
        Path.home(),
        Path("/tmp"),
        Path("/var"),
        Path("/etc"),
        Path("/usr"),
        Path("/opt"),
        Path("/private"),
    }
)


def _is_forbidden_path(path: Path) -> bool:
    return path in _FORBIDDEN_PATHS


class FilesystemMemory:
    """Filesystem-backed cross-step memory using the MCP filesystem server.

    Constructed once per serial loop / batch; passed to ``paper_flow``
    via the ``memory=`` kwarg. Each ``run_hooks()`` invocation returns
    a fresh ``MemoryRunHooks`` so per-run accumulator state is isolated;
    they share the per-instance ``asyncio.Lock`` that serialises
    ``runs.jsonl`` appends.

    The Agent only sees ``workspace/``; ``runs/`` and ``runs.jsonl``
    are out-of-reach.
    """

    def __init__(self, memory_dir: str | Path) -> None:
        self.memory_dir = Path(memory_dir).resolve()
        if _is_forbidden_path(self.memory_dir):
            raise ValueError(
                f"memory_dir {self.memory_dir!s} is on the protected list "
                f"({sorted(str(p) for p in _FORBIDDEN_PATHS)}); choose a "
                "dedicated subdirectory."
            )

        marker = self.memory_dir / _MARKER_NAME
        if (
            self.memory_dir.exists()
            and any(self.memory_dir.iterdir())
            and not marker.exists()
        ):
            raise ValueError(
                f"memory_dir {self.memory_dir!s} is non-empty and lacks the "
                f"{_MARKER_NAME!r} marker; refusing to manage it. Either "
                "point FilesystemMemory at an empty / fresh directory, or "
                f"add an empty {_MARKER_NAME!r} file to claim the directory."
            )

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)

        self.workspace = self.memory_dir / "workspace"
        for sub in ("notes", "items"):
            (self.workspace / sub).mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "runs").mkdir(parents=True, exist_ok=True)

        readme = self.workspace / "README.md"
        if not readme.exists():
            readme.write_text(_AGENT_README_TEXT, encoding="utf-8")

        self._archive_lock = asyncio.Lock()

    def tools(self) -> list[Tool]:
        return []

    def mcp_servers(self) -> list[MCPServer]:
        # The SDK default 5s timeout is too tight for the first run of
        # `npx -y @modelcontextprotocol/server-filesystem` (npm has to
        # resolve + extract the package on a cold cache, easily 10-20s
        # on slow connections). Bumping to 30s keeps subsequent
        # already-cached runs fast (~hundreds of ms to launch) while
        # making the first-run experience reliable.
        return [
            MCPServerStdio(
                name="quantmind_memory_fs",
                params={
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(self.workspace),
                    ],
                },
                client_session_timeout_seconds=30.0,
            )
        ]

    def run_hooks(self) -> RunHooks[Any] | None:
        return MemoryRunHooks(
            memory_dir=self.memory_dir,
            archive_lock=self._archive_lock,
        )

    async def reset(self) -> None:
        """Wipe ``workspace/`` and ``runs/`` plus ``runs.jsonl``.

        Destructive — irreversibly removes every file under those paths.
        The marker file is preserved so subsequent ``FilesystemMemory``
        constructions on the same directory remain allowed. Deletion
        errors are NOT silently swallowed: ``shutil.rmtree`` is called
        without ``ignore_errors`` so any underlying issue (permission,
        in-use file, ...) surfaces to the caller.
        """
        for sub_path in (self.workspace, self.memory_dir / "runs"):
            sub_resolved = sub_path.resolve()
            if not sub_resolved.is_relative_to(self.memory_dir):
                raise ValueError(
                    f"Refusing to delete outside memory_dir: {sub_resolved!s}"
                )
            if sub_resolved.exists():
                shutil.rmtree(sub_resolved)
        (self.memory_dir / "runs.jsonl").unlink(missing_ok=True)

        for sub in ("notes", "items"):
            (self.workspace / sub).mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "runs").mkdir(parents=True, exist_ok=True)
        readme = self.workspace / "README.md"
        if not readme.exists():
            readme.write_text(_AGENT_README_TEXT, encoding="utf-8")
