"""Tests for ``quantmind.mind.memory.filesystem``."""

import tempfile
import unittest
from pathlib import Path

from agents.mcp import MCPServerStdio

from quantmind.mind.memory._run_hooks import MemoryRunHooks
from quantmind.mind.memory.filesystem import _MARKER_NAME, FilesystemMemory


class FilesystemMemoryInitTests(unittest.TestCase):
    def test_creates_subdirs_and_readme(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            self.assertTrue((mem.memory_dir / _MARKER_NAME).exists())
            self.assertTrue(mem.workspace.is_dir())
            for sub in ("notes", "items"):
                self.assertTrue((mem.workspace / sub).is_dir())
            self.assertTrue((mem.memory_dir / "runs").is_dir())
            self.assertTrue((mem.workspace / "README.md").exists())

    def test_readme_only_seeded_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            (mem.workspace / "README.md").write_text("custom")
            FilesystemMemory(raw)
            self.assertEqual(
                (mem.workspace / "README.md").read_text(), "custom"
            )

    def test_rejects_non_empty_directory_without_marker(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw)
            (path / "user-data.txt").write_text("x")
            with self.assertRaises(ValueError):
                FilesystemMemory(path)

    def test_accepts_non_empty_directory_with_marker(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw)
            (path / _MARKER_NAME).touch()
            (path / "user-data.txt").write_text("x")
            mem = FilesystemMemory(path)
            self.assertTrue(mem.workspace.is_dir())
            self.assertTrue((path / "user-data.txt").exists())

    def test_deleted_marker_rejects_existing_memory_dir(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            (mem.memory_dir / _MARKER_NAME).unlink()
            with self.assertRaises(ValueError):
                FilesystemMemory(raw)

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
            self.assertEqual(servers[0].params.command, "npx")
            self.assertEqual(servers[0].params.args[-1], str(mem.workspace))

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
            (mem.workspace / "notes" / "n.md").write_text("x")
            (mem.workspace / "items" / "i.json").write_text("{}")
            (mem.memory_dir / "runs" / "r.json").write_text("{}")
            (mem.memory_dir / "runs.jsonl").write_text("{}\n")
            await mem.reset()
            for sub in ("notes", "items"):
                self.assertTrue((mem.workspace / sub).is_dir())
                self.assertEqual(list((mem.workspace / sub).iterdir()), [])
            self.assertTrue((mem.memory_dir / "runs").is_dir())
            self.assertEqual(list((mem.memory_dir / "runs").iterdir()), [])
            self.assertFalse((mem.memory_dir / "runs.jsonl").exists())
            self.assertTrue((mem.workspace / "README.md").exists())
            self.assertTrue((mem.memory_dir / _MARKER_NAME).exists())
