"""Tests for ``quantmind.flows._runner``."""

import tempfile
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from agents import RunHooks

from quantmind.configs import PaperFlowCfg
from quantmind.flows._runner import (
    _compose_hooks,
    _CompositeRunHooks,
    run_with_observability,
)
from quantmind.mind.memory import FilesystemMemory


class _RecordingHooks(RunHooks[Any]):
    """Test hook that records every lifecycle call on a shared list."""

    def __init__(self, label: str, log: list[tuple[str, str]]) -> None:
        self.label = label
        self.log = log

    async def on_llm_start(self, *_: Any, **__: Any) -> None:
        self.log.append((self.label, "on_llm_start"))

    async def on_llm_end(self, *_: Any, **__: Any) -> None:
        self.log.append((self.label, "on_llm_end"))

    async def on_agent_start(self, *_: Any, **__: Any) -> None:
        self.log.append((self.label, "on_agent_start"))

    async def on_agent_end(self, *_: Any, **__: Any) -> None:
        self.log.append((self.label, "on_agent_end"))

    async def on_handoff(self, *_: Any, **__: Any) -> None:
        self.log.append((self.label, "on_handoff"))

    async def on_tool_start(self, *_: Any, **__: Any) -> None:
        self.log.append((self.label, "on_tool_start"))

    async def on_tool_end(self, *_: Any, **__: Any) -> None:
        self.log.append((self.label, "on_tool_end"))


class _FakeRunResult:
    def __init__(self, output: str = "ok") -> None:
        self.final_output = output
        self.trace_id = "trace_xyz"


class ComposeHooksTests(unittest.TestCase):
    def test_empty_returns_none(self) -> None:
        self.assertIsNone(_compose_hooks([]))

    def test_single_returns_same_instance(self) -> None:
        hook = _RecordingHooks("a", [])
        self.assertIs(_compose_hooks([hook]), hook)

    def test_multiple_returns_composite(self) -> None:
        a = _RecordingHooks("a", [])
        b = _RecordingHooks("b", [])
        composed = _compose_hooks([a, b])
        self.assertIsInstance(composed, _CompositeRunHooks)


class CompositeRunHooksTests(unittest.IsolatedAsyncioTestCase):
    async def test_fan_out_in_registration_order(self) -> None:
        log: list[tuple[str, str]] = []
        a = _RecordingHooks("a", log)
        b = _RecordingHooks("b", log)
        composite = _CompositeRunHooks([a, b])
        await composite.on_llm_start()
        await composite.on_llm_end()
        await composite.on_agent_start()
        await composite.on_agent_end()
        await composite.on_handoff()
        await composite.on_tool_start()
        await composite.on_tool_end()
        for method in (
            "on_llm_start",
            "on_llm_end",
            "on_agent_start",
            "on_agent_end",
            "on_handoff",
            "on_tool_start",
            "on_tool_end",
        ):
            self.assertEqual(
                [entry for entry in log if entry[1] == method],
                [("a", method), ("b", method)],
            )

    async def test_earlier_hook_exception_short_circuits(self) -> None:
        class _Boom(RunHooks[Any]):
            async def on_llm_start(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("boom")

        log: list[tuple[str, str]] = []
        composite = _CompositeRunHooks([_Boom(), _RecordingHooks("b", log)])
        with self.assertRaises(RuntimeError):
            await composite.on_llm_start()
        self.assertEqual(log, [])


class RunWithObservabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_config_built_from_cfg(self) -> None:
        cfg = PaperFlowCfg(
            model="gpt-test",
            max_turns=7,
            workflow_name="custom-name",
            trace_metadata={"k": "v"},
            trace_include_sensitive_data=False,
            tracing_disabled=True,
        )
        agent = MagicMock(name="agent")
        agent.name = "paper_extractor"
        fake_result = MagicMock()
        fake_result.final_output = "OUT"
        with patch(
            "quantmind.flows._runner.Runner.run",
            new=AsyncMock(return_value=fake_result),
        ) as run_mock:
            out = await run_with_observability(
                agent,
                "prompt",
                cfg=cfg,
                memory=None,
                extra_run_hooks=[],
            )
        self.assertEqual(out, "OUT")
        run_mock.assert_awaited_once()
        call = run_mock.await_args
        self.assertIs(call.args[0], agent)
        self.assertEqual(call.args[1], "prompt")
        self.assertEqual(call.kwargs["max_turns"], 7)
        run_cfg = call.kwargs["run_config"]
        self.assertEqual(run_cfg.workflow_name, "custom-name")
        self.assertEqual(run_cfg.trace_metadata, {"k": "v"})
        self.assertFalse(run_cfg.trace_include_sensitive_data)
        self.assertTrue(run_cfg.tracing_disabled)
        self.assertIsNone(call.kwargs["hooks"])

    async def test_workflow_name_falls_back_to_agent_name(self) -> None:
        cfg = PaperFlowCfg()
        agent = MagicMock()
        agent.name = "paper_extractor"
        fake_result = MagicMock()
        fake_result.final_output = None
        with patch(
            "quantmind.flows._runner.Runner.run",
            new=AsyncMock(return_value=fake_result),
        ) as run_mock:
            await run_with_observability(
                agent, "x", cfg=cfg, memory=None, extra_run_hooks=[]
            )
        self.assertEqual(
            run_mock.await_args.kwargs["run_config"].workflow_name,
            "quantmind.paper_extractor",
        )

    async def test_extra_hooks_forwarded(self) -> None:
        cfg = PaperFlowCfg()
        agent = MagicMock()
        agent.name = "x"
        fake_result = MagicMock()
        fake_result.final_output = None
        hook = _RecordingHooks("a", [])
        with patch(
            "quantmind.flows._runner.Runner.run",
            new=AsyncMock(return_value=fake_result),
        ) as run_mock:
            await run_with_observability(
                agent,
                "x",
                cfg=cfg,
                memory=None,
                extra_run_hooks=[hook],
            )
        self.assertIs(run_mock.await_args.kwargs["hooks"], hook)


class RunnerMemoryWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_persist_invoked_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            cfg = PaperFlowCfg()
            agent = MagicMock()
            agent.name = "paper_extractor"
            with (
                patch(
                    "quantmind.flows._runner.Runner.run",
                    new=AsyncMock(return_value=_FakeRunResult("done")),
                ),
                patch(
                    "quantmind.mind.memory._run_hooks.write_run_record",
                    new=AsyncMock(),
                ) as mock_write,
            ):
                out = await run_with_observability(
                    agent,
                    "input",
                    cfg=cfg,
                    memory=mem,
                    extra_run_hooks=[],
                )
            self.assertEqual(out, "done")
            self.assertEqual(mock_write.await_count, 1)
            record = mock_write.await_args.args[1]
            self.assertIsNone(record.error)
            self.assertEqual(record.trace_id, "trace_xyz")

    async def test_persist_invoked_on_failure_with_error_string(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            cfg = PaperFlowCfg()
            agent = MagicMock()
            agent.name = "paper_extractor"
            boom = RuntimeError("boom")
            with (
                patch(
                    "quantmind.flows._runner.Runner.run",
                    new=AsyncMock(side_effect=boom),
                ),
                patch(
                    "quantmind.mind.memory._run_hooks.write_run_record",
                    new=AsyncMock(),
                ) as mock_write,
            ):
                with self.assertRaises(RuntimeError):
                    await run_with_observability(
                        agent,
                        "input",
                        cfg=cfg,
                        memory=mem,
                        extra_run_hooks=[],
                    )
            self.assertEqual(mock_write.await_count, 1)
            record = mock_write.await_args.args[1]
            self.assertEqual(record.error, "boom")

    async def test_archive_trajectory_false_skips_memory_hooks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mem = FilesystemMemory(raw)
            cfg = PaperFlowCfg(archive_trajectory=False)
            agent = MagicMock()
            agent.name = "paper_extractor"
            with (
                patch(
                    "quantmind.flows._runner.Runner.run",
                    new=AsyncMock(return_value=_FakeRunResult()),
                ),
                patch(
                    "quantmind.mind.memory._run_hooks.write_run_record",
                    new=AsyncMock(),
                ) as mock_write,
            ):
                await run_with_observability(
                    agent,
                    "input",
                    cfg=cfg,
                    memory=mem,
                    extra_run_hooks=[],
                )
            self.assertEqual(mock_write.await_count, 0)

    async def test_no_memory_skips_persist(self) -> None:
        cfg = PaperFlowCfg()
        agent = MagicMock()
        agent.name = "paper_extractor"
        with (
            patch(
                "quantmind.flows._runner.Runner.run",
                new=AsyncMock(return_value=_FakeRunResult()),
            ),
            patch(
                "quantmind.mind.memory._run_hooks.write_run_record",
                new=AsyncMock(),
            ) as mock_write,
        ):
            await run_with_observability(
                agent, "input", cfg=cfg, memory=None, extra_run_hooks=[]
            )
        self.assertEqual(mock_write.await_count, 0)
