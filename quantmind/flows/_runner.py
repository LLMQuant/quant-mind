"""Internal helpers shared by every flow function.

`run_with_observability` wraps `Runner.run` with `RunConfig` derived from
`BaseFlowCfg`, composes user-supplied `RunHooks` (the SDK accepts only a
single hooks instance per run), and leaves a no-op call site for the
PR6 trajectory archive. Flow modules call this instead of touching the
SDK directly so observability behaviour stays in one place.
"""

from typing import Any

from agents import Agent, RunConfig, RunHooks, Runner

from quantmind.configs import BaseFlowCfg
from quantmind.flows._usage import record_usage


async def run_with_observability(
    agent: Agent[Any],
    input: str | list[Any],
    *,
    cfg: BaseFlowCfg,
    memory: object | None = None,
    extra_run_hooks: list[RunHooks[Any]],
) -> Any:
    """Build `RunConfig` + composed hooks, run the agent, return final output.

    Args:
        agent: The Agents SDK ``Agent`` to invoke.
        input: Prompt string or pre-built input items.
        cfg: Flow configuration. Tracing fields and ``max_turns`` are
            forwarded to the SDK; ``workflow_name`` falls back to
            ``"quantmind.<agent.name>"`` when unset.
        memory: PR6 ``Memory`` placeholder. Currently unused at runtime;
            the value is forwarded to the trajectory-archive stub so PR6
            can wire it in without changing call sites.
        extra_run_hooks: User-supplied hooks. Composed with any
            memory-derived hooks (none in PR5) into a single
            ``RunHooks`` instance.

    Returns:
        ``RunResult.final_output`` typed by the agent's ``output_type``.
    """
    workflow_name = cfg.workflow_name or f"quantmind.{agent.name}"
    run_cfg = RunConfig(
        workflow_name=workflow_name,
        trace_metadata=cfg.trace_metadata,
        trace_include_sensitive_data=cfg.trace_include_sensitive_data,
        tracing_disabled=cfg.tracing_disabled,
    )
    hooks = _compose_hooks(_collect_hooks(memory, extra_run_hooks))
    result = await Runner.run(
        agent,
        input,
        run_config=run_cfg,
        hooks=hooks,
        max_turns=cfg.max_turns,
    )
    # No-op unless a `usage_scope` is active (e.g. inside `batch_run`);
    # keeps every existing caller's behaviour unchanged.
    record_usage(result.context_wrapper.usage)
    _archive_run_artifacts(cfg, memory, result)
    return result.final_output


def _collect_hooks(
    memory: object | None,
    extras: list[RunHooks[Any]],
) -> list[RunHooks[Any]]:
    """Return hooks in run order: memory hooks first (PR6), then extras."""
    hooks: list[RunHooks[Any]] = []
    # PR6 will append `memory.run_hooks()` here when `memory` exposes the
    # `Memory` Protocol. PR5 keeps `memory` opaque and contributes no hooks.
    del memory
    hooks.extend(extras)
    return hooks


def _compose_hooks(
    hooks: list[RunHooks[Any]],
) -> RunHooks[Any] | None:
    """Merge multiple `RunHooks` into one (the SDK takes a single instance)."""
    if not hooks:
        return None
    if len(hooks) == 1:
        return hooks[0]
    return _CompositeRunHooks(hooks)


class _CompositeRunHooks(RunHooks[Any]):
    """Fan out every lifecycle method to each wrapped hook in order.

    Exceptions from earlier hooks short-circuit the rest by design — hooks
    are integral to the run, not best-effort. PR6's archive hook should
    catch its own exceptions internally if it wants resilience.
    """

    def __init__(self, inner: list[RunHooks[Any]]) -> None:
        self._inner = list(inner)

    async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        for h in self._inner:
            await h.on_llm_start(*args, **kwargs)

    async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        for h in self._inner:
            await h.on_llm_end(*args, **kwargs)

    async def on_agent_start(self, *args: Any, **kwargs: Any) -> None:
        for h in self._inner:
            await h.on_agent_start(*args, **kwargs)

    async def on_agent_end(self, *args: Any, **kwargs: Any) -> None:
        for h in self._inner:
            await h.on_agent_end(*args, **kwargs)

    async def on_handoff(self, *args: Any, **kwargs: Any) -> None:
        for h in self._inner:
            await h.on_handoff(*args, **kwargs)

    async def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        for h in self._inner:
            await h.on_tool_start(*args, **kwargs)

    async def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        for h in self._inner:
            await h.on_tool_end(*args, **kwargs)


def _archive_run_artifacts(
    cfg: BaseFlowCfg,
    memory: object | None,
    result: Any,
) -> None:
    """No-op stub. PR6 writes a trajectory record under ``<memory_dir>/runs/``.

    Kept as a real call site (rather than commented-out) so PR6 changes
    one function body, not the runner's public path.
    """
    del cfg, memory, result
    return None
