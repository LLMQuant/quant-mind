"""05 — Same flow, DeepSeek model. Only the model string changes.

This is the headline ergonomic claim of QuantMind's provider layer:
to point the flow at a different LLM provider, you change ONE thing
on the cfg:

    PaperFlowCfg(model="deepseek-chat")     # vs "gpt-4o"

QuantMind's provider auto-detector (``quantmind.flows._providers``)
reads the model-name prefix, looks up the right SDK client class
(Chat Completions for DeepSeek, Responses for OpenAI), resolves the
API key from the matching env var (``DEEPSEEK_API_KEY``), forces
``tracing_disabled=True`` so the SDK does not try to upload traces
to ``platform.openai.com``, and hands a fully-built Model object to
the Agent.

Adding a new provider in the future means appending one row to
``_PROVIDERS`` — this example file does not change.

Prerequisites:

- ``bash scripts/setup.sh`` already run (creates .venv, installs deps,
  audits node/npx).
- ``export DEEPSEEK_API_KEY="sk-..."`` in your shell (or .env).
- Node.js + ``npx`` on PATH for ``FilesystemMemory`` to launch its
  MCP filesystem server. ``scripts/check_system_deps.py`` confirms
  this for you.

Run:
    python examples/memory/05_deepseek.py
"""

import asyncio
from pathlib import Path

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import RawText
from quantmind.flows import paper_flow
from quantmind.mind.memory import FilesystemMemory


async def main() -> None:
    mem = FilesystemMemory(Path("./.qm-memory"))

    # *** This is the entire provider switch. ***
    # No `set_default_openai_client`, no `AsyncOpenAI(...)` boilerplate,
    # no `set_tracing_disabled(True)`. The flow handles it internally.
    cfg = PaperFlowCfg(model="deepseek-chat")

    paper = await paper_flow(
        RawText(
            text=(
                "# Toy paper for DeepSeek smoke test\n\n"
                "Title: Cross-sectional momentum, simplified\n"
                "Author: Demo\n\n"
                "Body: We illustrate momentum on a 3-stock universe."
            )
        ),
        cfg=cfg,
        memory=mem,
    )
    print(f"Extracted paper: {paper.title!r}")

    # Sanity-check that the trajectory record landed and captured
    # provider-correct token usage.
    runs = sorted((mem.memory_dir / "runs").glob("*.json"))
    print(f"\nTrajectory file: {runs[-1]}")


if __name__ == "__main__":
    asyncio.run(main())
