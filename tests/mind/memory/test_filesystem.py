"""Tests for ``quantmind.mind.memory.filesystem``."""

import tempfile
import unittest
from pathlib import Path

from agents.mcp import MCPServerStdio

from quantmind.mind.memory._run_hooks import MemoryRunHooks
from quantmind.mind.memory.filesystem import FilesystemMemory


class FilesystemMemoryInitTests(unittest.TestCase):
    def test_creates_subdirs_and_readme(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            for sub in ("notes", "items", "runs"):
                self.assertTrue((mem.memory_dir / sub).is_dir())
            self.assertTrue((mem.memory_dir / "README.md").exists())

    def test_readme_only_seeded_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            (mem.memory_dir / "README.md").write_text("custom")
            FilesystemMemory(raw)
            self.assertEqual(
                (mem.memory_dir / "README.md").read_text(), "custom"
            )

    def test_rejects_root_path(self) -> None:
        with self.assertRaises(ValueError):
            FilesystemMemory("/")

    def test_rejects_home_path(self) -> None:
        with self.assertRaises(ValueError):
            FilesystemMemory(str(Path.home()))


class FilesystemMemoryMethodsTests(unittest.TestCase):
    def test_tools_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            self.assertEqual(mem.tools(), [])

    def test_mcp_servers_returns_stdio_with_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            servers = mem.mcp_servers()
            self.assertEqual(len(servers), 1)
            self.assertIsInstance(servers[0], MCPServerStdio)

    def test_mcp_servers_returns_fresh_instance_each_call(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            self.assertIsNot(mem.mcp_servers()[0], mem.mcp_servers()[0])

    def test_run_hooks_returns_memory_run_hooks_with_lock(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            h1 = mem.run_hooks()
            h2 = mem.run_hooks()
            self.assertIsInstance(h1, MemoryRunHooks)
            self.assertIsNot(h1, h2)
            assert isinstance(h1, MemoryRunHooks)
            assert isinstance(h2, MemoryRunHooks)
            self.assertIs(h1._archive_lock, h2._archive_lock)


class FilesystemMemoryResetTests(unittest.IsolatedAsyncioTestCase):
    async def test_reset_wipes_subdirs_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            (mem.memory_dir / "notes" / "n.md").write_text("x")
            (mem.memory_dir / "items" / "i.json").write_text("{}")
            (mem.memory_dir / "runs" / "r.json").write_text("{}")
            (mem.memory_dir / "runs.jsonl").write_text("{}\n")
            await mem.reset()
            for sub in ("notes", "items", "runs"):
                self.assertTrue((mem.memory_dir / sub).is_dir())
                self.assertEqual(list((mem.memory_dir / sub).iterdir()), [])
            self.assertFalse((mem.memory_dir / "runs.jsonl").exists())
            self.assertTrue((mem.memory_dir / "README.md").exists())
