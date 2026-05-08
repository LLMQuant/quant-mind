"""Internal helpers shared by every flow function.

`run_with_observability` wraps `Runner.run` with `RunConfig` derived from
`BaseFlowCfg`, composes user-supplied `RunHooks` (the SDK accepts only a
single hooks instance per run), and orchestrates the
`MemoryRunHooks.persist()` call in `finally` so failed runs still
produce a trajectory record.
"""

from typing import Any

from agents import Agent, RunConfig, RunHooks, Runner

from quantmind.configs import BaseFlowCfg
from quantmind.mind.memory import Memory, MemoryRunHooks


async def run_with_observability(
    agent: Agent[Any],
    input: str | list[Any],
    *,
    cfg: BaseFlowCfg,
    memory: Memory | None = None,
    extra_run_hooks: list[RunHooks[Any]],
) -> Any:
    """Build `RunConfig` + composed hooks, run the agent, return final output.

    Args:
        agent: The Agents SDK ``Agent`` to invoke.
        input: Prompt string or pre-built input items.
        cfg: Flow configuration. Tracing fields and ``max_turns`` are
            forwarded to the SDK; ``workflow_name`` falls back to
            ``"quantmind.<agent.name>"`` when unset.
        memory: Optional ``Memory`` implementation. When set and
            ``cfg.archive_trajectory`` is True, ``memory.run_hooks()``
            participates in the run and ``MemoryRunHooks.persist()``
            is invoked in ``finally`` (so failures archive too).
        extra_run_hooks: User-supplied hooks composed after the memory
            hook.

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

    memory_hooks: MemoryRunHooks | None = None
    hooks_list: list[RunHooks[Any]] = []
    if memory is not None and cfg.archive_trajectory:
        h = memory.run_hooks()
        if h is not None:
            hooks_list.append(h)
            if isinstance(h, MemoryRunHooks):
                memory_hooks = h
    hooks_list.extend(extra_run_hooks)
    composed = _compose_hooks(hooks_list)

    result: Any = None
    error: BaseException | None = None
    try:
        result = await Runner.run(
            agent,
            input,
            run_config=run_cfg,
            hooks=composed,
            max_turns=cfg.max_turns,
        )
        return result.final_output
    except BaseException as exc:
        error = exc
        raise
    finally:
        if memory_hooks is not None:
            await memory_hooks.persist(
                workflow_name=workflow_name,
                result=result,
                error=error,
                input_payload=input,
            )


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
    are integral to the run, not best-effort.
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
