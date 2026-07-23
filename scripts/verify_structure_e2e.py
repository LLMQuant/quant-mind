#!/usr/bin/env python3
"""Run the bounded live structure-build + agentic-retrieve slice.

Offline tests mock the SDK model, so they never exercise a real structured-output
draft call or a real agentic traversal loop. This bounded live slice builds a
self-contained ``PaperStructureTree`` from the golden fixture PDF with
``PaperFlow`` and runs a real ``AgenticRetriever`` traversal over it. It uses
two OpenRouter models: DeepSeek V4 Flash exercises JSON-object compatibility,
while GPT-5.6 Luna is the regression baseline.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from quantmind.configs import PaperStructureCfg, RetrievalCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import PaperFlow
from quantmind.mind import AgenticRetriever

_PDF = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/paper/golden/paper.pdf"
)
_QUESTION = "What is the main method and its main limitation?"
_MODELS = (
    "litellm/openrouter/deepseek/deepseek-v4-flash",
    "litellm/openrouter/openai/gpt-5.6-luna",
)


async def _run_slice(model: str) -> dict[str, Any]:
    # Build: config-bound flow and self-contained tree.
    tree = await PaperFlow(
        PaperStructureCfg(
            model=model,
            timeout_seconds=240,
            tracing_disabled=True,
        )
    ).build(LocalFilePath(path=_PDF))
    root_pages = {c.page for c in tree.root().citations if c.page}
    leaf_contents = [
        node.content
        for node in tree.nodes.values()
        if not node.children_ids and node.content
    ]

    # Retrieve: real agentic traversal, library-free, over the built tree.
    evidence = await AgenticRetriever(
        RetrievalCfg(
            model=model,
            timeout_seconds=240,
            tracing_disabled=True,
        )
    ).retrieve(tree, _QUESTION)

    return {
        "model": model,
        "node_count": len(tree.nodes),
        "root_page_span": sorted(root_pages),
        "leaf_content_count": len(leaf_contents),
        "evidence_titles": [item.title for item in evidence],
        "evidence_all_have_content": all(item.content for item in evidence),
        "evidence_pages": sorted(
            {c.page for item in evidence for c in item.citations if c.page}
        ),
    }


def _passed(s: dict[str, Any]) -> bool:
    return bool(
        s["node_count"] >= 2
        and s["root_page_span"]
        and s["leaf_content_count"] >= 1
        and s["evidence_titles"]
        and s["evidence_all_have_content"]
    )


async def main() -> int:
    """Run and report the timeout-bounded live structure slice."""
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        print("[FAIL] structure-retrieval: OPENROUTER_API_KEY is not set")
        return 1
    passed = True
    for model in _MODELS:
        try:
            snapshot = await asyncio.wait_for(_run_slice(model), timeout=480)
        except Exception as exc:
            print(
                f"[FAIL] structure-retrieval: model={model} "
                f"{type(exc).__name__}: {exc}"
            )
            passed = False
            continue

        state = "PASS" if _passed(snapshot) else "FAIL"
        print(
            f"[{state}] structure-retrieval: model={snapshot['model']} "
            f"nodes={snapshot['node_count']} "
            f"root_pages={snapshot['root_page_span']} "
            f"leaves_with_content={snapshot['leaf_content_count']} "
            f"evidence={snapshot['evidence_titles']} "
            f"evidence_pages={snapshot['evidence_pages']}"
        )
        passed = passed and state == "PASS"
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
