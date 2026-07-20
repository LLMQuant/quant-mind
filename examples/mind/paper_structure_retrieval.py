"""Build a self-contained structure tree and retrieve evidence in memory.

The core path needs no library: ``PaperFlow`` opens the PDF (fetch + parse
once), ``build_structure`` returns a self-contained ``PaperStructureTree`` whose
leaf nodes carry their own page-cited text, and ``retrieve`` reasons over that
tree value and returns evidence with the content already in it.

The clearly-labeled OPTIONAL section then shows dump/load symmetry: a library
stores the tree and reopens it as an identical self-contained value, over which
``retrieve`` behaves exactly the same. It is not needed to retrieve.
"""

import asyncio
import sys
from pathlib import Path

from quantmind.configs import PaperStructureCfg, RetrievalCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import PaperFlow
from quantmind.knowledge import PaperStructureTree
from quantmind.mind import retrieve

_QUESTION = "What are the main method and limitations?"


async def main(pdf_path: Path) -> None:
    """Build a structure tree and retrieve from it in memory (no library)."""
    paper = await PaperFlow.open(LocalFilePath(path=pdf_path))
    tree = await paper.build_structure(
        cfg=PaperStructureCfg(model="gpt-4o-mini")
    )

    evidence = await retrieve(
        tree,
        _QUESTION,
        cfg=RetrievalCfg(model="gpt-4o-mini", grain="agentic"),
    )
    for item in evidence:
        print(item.title, "—", item.content[:500])

    await _optional_persist_reopen(paper, tree)


async def _optional_persist_reopen(
    paper: PaperFlow,
    tree: PaperStructureTree,
) -> None:
    """OPTIONAL: dump the tree to a library and reopen an identical value.

    This section is not required to retrieve — the tree built above is already
    self-contained. It exists only to demonstrate that
    ``put_paper_structure_tree`` / ``open_structure`` round-trip to the same
    value, over which ``retrieve`` behaves identically. Delete it if you only
    need in-memory retrieval.
    """
    from quantmind.library import LocalKnowledgeLibrary

    library = await LocalKnowledgeLibrary.open(
        ":memory:",
        embedding_model="text-embedding-3-small",
    )
    try:
        await library.put_paper_structure_tree(paper.source, tree)
        reopened = await library.open_structure(tree.id)  # identical value
        evidence = await retrieve(
            reopened,
            _QUESTION,
            cfg=RetrievalCfg(model="gpt-4o-mini", grain="agentic"),
        )
        for item in evidence:
            print(item.title, "—", item.content[:500])
    finally:
        await library.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python examples/mind/paper_structure_retrieval.py paper.pdf"
        )
    asyncio.run(main(Path(sys.argv[1])))
