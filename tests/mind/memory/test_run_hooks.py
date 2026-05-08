"""Tests for ``quantmind.mind.memory._run_hooks``."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from quantmind.mind.memory._run_hooks import MemoryRunHooks


def _make_hooks(tmpdir: Path) -> MemoryRunHooks:
    (tmpdir / "runs").mkdir(parents=True, exist_ok=True)
    return MemoryRunHooks(memory_dir=tmpdir, archive_lock=asyncio.Lock())


class MemoryRunHooksLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_on_agent_start_records_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            hooks = _make_hooks(Path(raw))
            agent = SimpleNamespace(
                name="paper_extractor",
                model="gpt-4o",
                instructions="extract papers",
            )
            await hooks.on_agent_start(SimpleNamespace(), agent)
            self.assertEqual(hooks._agent_name, "paper_extractor")
            self.assertEqual(hooks._agent_model, "gpt-4o")
            self.assertEqual(len(hooks._instructions_hash), 16)

    async def test_on_llm_end_accumulates_call(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            hooks = _make_hooks(Path(raw))
            await hooks.on_agent_start(
                SimpleNamespace(),
                SimpleNamespace(name="a", model="gpt-4o", instructions=""),
            )
            await hooks.on_llm_start()
            response = SimpleNamespace(
                usage=SimpleNamespace(input_tokens=42, output_tokens=11)
            )
            await hooks.on_llm_end(
                SimpleNamespace(),
                SimpleNamespace(model="gpt-4o"),
                response,
            )
            self.assertEqual(len(hooks._llm_calls), 1)
            self.assertEqual(hooks._llm_calls[0]["tokens_in"], 42)
            self.assertEqual(hooks._llm_calls[0]["tokens_out"], 11)

    async def test_on_llm_end_handles_missing_usage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            hooks = _make_hooks(Path(raw))
            await hooks.on_llm_start()
            await hooks.on_llm_end(
                SimpleNamespace(),
                SimpleNamespace(model="gpt-4o"),
                SimpleNamespace(usage=None),
            )
            self.assertEqual(hooks._llm_calls[0]["tokens_in"], 0)
            self.assertEqual(hooks._llm_calls[0]["tokens_out"], 0)

    async def test_on_tool_pair_records_call(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            hooks = _make_hooks(Path(raw))
            tool = SimpleNamespace(name="read_file")
            await hooks.on_tool_start(
                SimpleNamespace(), SimpleNamespace(), tool
            )
            await hooks.on_tool_end(
                SimpleNamespace(),
                SimpleNamespace(),
                tool,
                "file contents",
            )
            self.assertEqual(len(hooks._tool_calls), 1)
            self.assertEqual(hooks._tool_calls[0]["name"], "read_file")
            self.assertGreaterEqual(hooks._tool_calls[0]["duration_s"], 0.0)

    async def test_on_agent_end_truncates_long_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            hooks = _make_hooks(Path(raw))
            big = "x" * 1000
            await hooks.on_agent_end(SimpleNamespace(), SimpleNamespace(), big)
            self.assertTrue(hooks._output_summary.endswith("[truncated]"))
            self.assertLessEqual(len(hooks._output_summary), 500)


class MemoryRunHooksPersistTests(unittest.IsolatedAsyncioTestCase):
    async def test_persist_success_writes_record_with_no_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem_dir = Path(raw)
            hooks = _make_hooks(mem_dir)
            await hooks.on_agent_start(
                SimpleNamespace(),
                SimpleNamespace(name="a", model="gpt-4o", instructions="x"),
            )
            await hooks.on_agent_end(
                SimpleNamespace(), SimpleNamespace(), "out"
            )
            with patch(
                "quantmind.mind.memory._run_hooks.write_run_record",
                new=AsyncMock(),
            ) as mock_write:
                await hooks.persist(
                    workflow_name="quantmind.paper_flow",
                    result=SimpleNamespace(trace_id="trace_xx"),
                    error=None,
                    input_payload="hello",
                )
            args, _kwargs = mock_write.call_args
            record = args[1]
            self.assertEqual(record.workflow_name, "quantmind.paper_flow")
            self.assertEqual(record.trace_id, "trace_xx")
            self.assertIsNone(record.error)

    async def test_persist_error_path_records_exception_string(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem_dir = Path(raw)
            hooks = _make_hooks(mem_dir)
            with patch(
                "quantmind.mind.memory._run_hooks.write_run_record",
                new=AsyncMock(),
            ) as mock_write:
                await hooks.persist(
                    workflow_name="quantmind.paper_flow",
                    result=None,
                    error=ValueError("boom"),
                    input_payload="hi",
                )
            record = mock_write.call_args.args[1]
            self.assertEqual(record.error, "boom")
