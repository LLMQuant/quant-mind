"""Tests for ``quantmind.flows._runner``."""

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from agents import RunHooks

from quantmind.configs import PaperSemanticCfg
from quantmind.flows._runner import (
    _archive_run_artifacts,
    _collect_hooks,
    _compose_hooks,
    _CompositeRunHooks,
    run_with_observability,
)


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
        # Each method fires for both hooks in registration order.
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


class CollectHooksTests(unittest.TestCase):
    def test_memory_contributes_nothing_in_pr5(self) -> None:
        extra = _RecordingHooks("a", [])
        # PR5: memory is forwarded but unused.
        self.assertEqual(_collect_hooks(None, [extra]), [extra])
        self.assertEqual(_collect_hooks(object(), [extra]), [extra])

    def test_no_extras_returns_empty(self) -> None:
        self.assertEqual(_collect_hooks(None, []), [])


class ArchiveStubTests(unittest.TestCase):
    def test_archive_is_no_op(self) -> None:
        cfg = PaperSemanticCfg()
        result = MagicMock()
        # Must not raise, must return None, must not touch result.
        self.assertIsNone(_archive_run_artifacts(cfg, None, result))
        result.assert_not_called()


class RunWithObservabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_config_built_from_cfg(self) -> None:
        cfg = PaperSemanticCfg(
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
        # No hooks supplied -> Runner.run sees None.
        self.assertIsNone(call.kwargs["hooks"])

    async def test_workflow_name_falls_back_to_agent_name(self) -> None:
        cfg = PaperSemanticCfg()  # workflow_name = None
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
        cfg = PaperSemanticCfg()
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
                memory=object(),  # PR6 placeholder
                extra_run_hooks=[hook],
            )
        # Single hook -> passed through as-is, not wrapped in composite.
        self.assertIs(run_mock.await_args.kwargs["hooks"], hook)
