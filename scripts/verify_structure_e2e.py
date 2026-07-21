#!/usr/bin/env python3
"""Run the bounded live structure-build + agentic-retrieve slice.

Offline tests mock the SDK model, so they never exercise a real structured-output
draft call or a real agentic traversal loop. This bounded live slice does: it
builds a self-contained ``PaperStructureTree`` from the golden fixture PDF with
``PaperFlow``, dumps and reopens it through the library unchanged, and runs a
real ``AgenticRetriever`` traversal — all under the default (config) model.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from quantmind.configs import PaperStructureCfg, RetrievalCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import PaperFlow
from quantmind.library import LocalKnowledgeLibrary
from quantmind.mind import AgenticRetriever

_PDF = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/paper/golden/paper.pdf"
)
_EMBEDDING_MODEL = "text-embedding-3-small"
_QUESTION = "What is the main method and its main limitation?"


async def _run_slice() -> dict[str, Any]:
    # Build: config-bound flow, default model, self-contained tree.
    tree = await PaperFlow(PaperStructureCfg(timeout_seconds=240)).build(
        LocalFilePath(path=_PDF)
    )
    root_pages = {c.page for c in tree.root().citations if c.page}
    leaf_contents = [
        node.content
        for node in tree.nodes.values()
        if not node.children_ids and node.content
    ]

    # Persist standalone (no source, no chunk set) and reopen unchanged.
    library = await LocalKnowledgeLibrary.open(
        ":memory:", embedding_model=_EMBEDDING_MODEL
    )
    try:
        await library.put(tree)
        reopened = await library.open_structure(tree.id)
    finally:
        await library.close()

    # Retrieve: real agentic traversal, library-free, over the reopened tree.
    evidence = await AgenticRetriever(
        RetrievalCfg(timeout_seconds=240)
    ).retrieve(reopened, _QUESTION)

    return {
        "node_count": len(tree.nodes),
        "root_page_span": sorted(root_pages),
        "leaf_content_count": len(leaf_contents),
        "reopened_identical": (
            reopened.id == tree.id
            and reopened.content_hash == tree.content_hash
        ),
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
        and s["reopened_identical"]
        and s["evidence_titles"]
        and s["evidence_all_have_content"]
    )


async def main() -> int:
    """Run and report the timeout-bounded live structure slice."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("[FAIL] structure-retrieval: OPENAI_API_KEY is not set")
        return 1
    try:
        snapshot = await asyncio.wait_for(_run_slice(), timeout=480)
    except Exception as exc:
        print(f"[FAIL] structure-retrieval: {type(exc).__name__}: {exc}")
        return 1

    state = "PASS" if _passed(snapshot) else "FAIL"
    print(
        f"[{state}] structure-retrieval: nodes={snapshot['node_count']} "
        f"root_pages={snapshot['root_page_span']} "
        f"leaves_with_content={snapshot['leaf_content_count']} "
        f"reopened_identical={snapshot['reopened_identical']} "
        f"evidence={snapshot['evidence_titles']} "
        f"evidence_pages={snapshot['evidence_pages']}"
    )
    return 0 if state == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
