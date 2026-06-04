"""04 — Custom RunHooks composing with MemoryRunHooks.

``paper_flow`` accepts ``extra_run_hooks=[...]`` so you can attach
your own ``RunHooks`` (logger, metrics emitter, slack ping, ...)
alongside the built-in ``MemoryRunHooks`` that ``FilesystemMemory``
contributes. The runner composes them all into one fan-out hook that
fires every lifecycle event for every registered hook in registration
order.

What to look for:

- ``[memory]`` lines printed by the built-in MemoryRunHooks aren't
  visible (it accumulates silently, then writes ``runs/<id>.json``
  in ``finally``).
- ``[my-logger]`` lines below come from ``ConsoleLoggerHooks``, the
  custom hook we plug in here.

Prerequisites: OPENAI_API_KEY, Node.js + npx.

Run:
    python examples/memory/04_custom_run_hooks.py
"""

import asyncio
from pathlib import Path
from typing import Any

from agents import RunHooks

from quantmind.configs.paper import RawText
from quantmind.flows import paper_flow
from quantmind.mind.memory import FilesystemMemory


class ConsoleLoggerHooks(RunHooks[Any]):
    """Tiny example hook that prints lifecycle events to stdout."""

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        print(
            f"[my-logger] agent start name={agent.name!r} model={agent.model!r}"
        )

    async def on_llm_start(self, *_: Any, **__: Any) -> None:
        print("[my-logger] llm start")

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            print(
                f"[my-logger] llm end tokens_in={getattr(usage, 'input_tokens', 0)} "
                f"tokens_out={getattr(usage, 'output_tokens', 0)}"
            )
        else:
            print("[my-logger] llm end (no usage info)")

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        print(f"[my-logger] tool start name={getattr(tool, 'name', '?')!r}")

    async def on_tool_end(
        self, context: Any, agent: Any, tool: Any, result: Any
    ) -> None:
        snippet = (
            (str(result)[:60] + "...") if len(str(result)) > 60 else result
        )
        print(
            f"[my-logger] tool end name={getattr(tool, 'name', '?')!r} "
            f"result={snippet!r}"
        )

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        print(f"[my-logger] agent end output_type={type(output).__name__}")

    async def on_handoff(
        self, context: Any, from_agent: Any, to_agent: Any
    ) -> None:
        print(
            f"[my-logger] handoff "
            f"{getattr(from_agent, 'name', '?')!r} -> "
            f"{getattr(to_agent, 'name', '?')!r}"
        )


async def main() -> None:
    mem = FilesystemMemory(Path("./.qm-memory"))

    paper = await paper_flow(
        RawText(
            text=(
                "# Toy paper\n\nTitle: Custom-hook demo\n\n"
                "Body: Show that user RunHooks fire alongside MemoryRunHooks."
            )
        ),
        memory=mem,
        extra_run_hooks=[ConsoleLoggerHooks()],
    )
    print(f"\nExtracted: {paper.nodes[paper.root_node_id].title!r}")
    print(
        f"\nTrajectory file: "
        f"{sorted((mem.memory_dir / 'runs').glob('*.json'))[-1]}"
    )


if __name__ == "__main__":
    asyncio.run(main())
