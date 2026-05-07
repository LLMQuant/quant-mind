"""Tests for ``quantmind.mind.memory._protocol``."""

import unittest
from typing import Any

from agents import RunHooks, Tool
from agents.mcp import MCPServer

from quantmind.mind.memory import Memory


class _CompleteStub:
    def tools(self) -> list[Tool]:
        return []

    def mcp_servers(self) -> list[MCPServer]:
        return []

    def run_hooks(self) -> RunHooks[Any] | None:
        return None

    async def reset(self) -> None:
        return None


class _MissingMethodStub:
    def tools(self) -> list[Tool]:
        return []


class MemoryProtocolTests(unittest.TestCase):
    def test_runtime_checkable_accepts_complete_stub(self) -> None:
        self.assertIsInstance(_CompleteStub(), Memory)

    def test_runtime_checkable_rejects_incomplete_stub(self) -> None:
        self.assertNotIsInstance(_MissingMethodStub(), Memory)
