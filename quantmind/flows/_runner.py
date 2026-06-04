"""Internal helpers shared by every flow function.

`run_with_observability` wraps `Runner.run` with `RunConfig` derived from
`BaseFlowCfg`, composes user-supplied `RunHooks` (the SDK accepts only a
single hooks instance per run), and orchestrates the trajectory archive
in `finally` so failed runs still produce a `RunRecord` (with `error`
set). The persistence call is guarded so its own failure never masks
the original run exception.
"""

from contextlib import AsyncExitStack
from typing import Any

from agents import Agent, RunConfig, RunHooks, Runner

from quantmind.configs import BaseFlowCfg
from quantmind.mind.memory import Memory
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


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
            participates in the run; if the returned hook exposes a
            ``persist(...)`` coroutine (duck-typed) it is invoked in
            ``finally`` so failures archive too.
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

    persistable_hook: Any = None
    hooks_list: list[RunHooks[Any]] = []
    if memory is not None and cfg.archive_trajectory:
        h = memory.run_hooks()
        if h is not None:
            hooks_list.append(h)
            if callable(getattr(h, "persist", None)):
                persistable_hook = h
    hooks_list.extend(extra_run_hooks)
    composed = _compose_hooks(hooks_list)

    result: Any = None
    error: BaseException | None = None
    try:
        # MCP servers are async context managers; the SDK does NOT
        # auto-connect them — list_tools() raises if connect() was
        # never called. We enter them here so the agent's Runner.run
        # sees connected servers, and exit them on the way out so the
        # ``npx`` subprocesses are reaped even on exception.
        async with AsyncExitStack() as mcp_stack:
            for server in getattr(agent, "mcp_servers", []) or []:
                await mcp_stack.enter_async_context(server)
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
        if persistable_hook is not None:
            try:
                await persistable_hook.persist(
                    workflow_name=workflow_name,
                    result=result,
                    error=error,
                    input_payload=input,
                )
            except BaseException as persist_exc:
                # Never let an archive failure mask an in-flight run
                # exception (which is the user's actual problem). When
                # the run succeeded, surface the archive failure normally.
                if error is None:
                    raise
                logger.warning(
                    "Failed to persist trajectory record after run failure "
                    "(%s); original error preserved.",
                    persist_exc,
                    exc_info=True,
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
